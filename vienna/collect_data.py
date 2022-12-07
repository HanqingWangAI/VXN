import gc
import os
import random
import warnings
from collections import defaultdict
from gym import Space
from habitat import Config, logger

import json
import gzip
import lmdb
import msgpack_numpy
import numpy as np
import torch
import tqdm
from habitat import logger
from habitat.datasets.utils import VocabDict
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.base_trainer import BaseTrainer


from vienna.common.aux_losses import AuxLosses
from vienna.common.base_il_trainer import BaseVLNCETrainer
from vienna.common.env_utils import construct_envs
from vienna.common.utils import extract_instruction_tokens
from vienna.models.CLEnc import CLEncAgent

from vienna.config.default import add_pano_sensors_to_config

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

import pickle
import time

class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def read_vocab(path):
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab

def collate_fn(batch):
    """Each sample in batch: (
        obs,
        prev_actions,
        oracle_actions,
        inflec_weight,
    )
    """
    res = defaultdict(list)

    for item_dict in batch:
        for key in item_dict:
            res[key].append(item_dict[key])

    return res


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]


class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        key_path,
        instructions_path,
        lmdb_map_size=1e9,
        batch_size=1,
        neg_num_vocab=200,
        neg_num_views=2000
    ):
        super().__init__()
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 5
        self._preload = []
        self.batch_size = batch_size
        self.neg_num_vocab = neg_num_vocab
        self.neg_num_views = neg_num_views
        self.instructions_path = instructions_path

        with open(self.instructions_path, 'rb') as fp:
            self.instruction_queries = pickle.load(fp) # {traj_id: [ins_list_1, ins_list_2, ...]}

        
        with open(key_path, 'rb') as fp:
            key_set = pickle.load(fp)
            self.key_list = [_.decode() for _ in key_set]
        
        self.tid2kid = {}
        self.tid_list = []
        for key in self.key_list:
            eid, tid = key.split('_')
            if not tid in self.tid2kid:
                self.tid2kid[tid] = key
                self.tid_list.append(tid)

        with gzip.open('data/datasets/R2R_VLNCE_v1-2_preprocessed/train/train.json.gz') as fp:
            deserialized = json.loads(fp.read())
        
        self.instruction_vocab = VocabDict(word_list=deserialized["instruction_vocab"]["word_list"])

        include_word_list = read_vocab('data/datasets/R2R_VLNCE_v1-2_preprocessed/include_vocab.txt')
        self.noun_set = set()

        for w in include_word_list:
            idx = self.instruction_vocab.word2idx(w)
            if idx != self.instruction_vocab.UNK_INDEX:
                self.noun_set.add(idx)

        self.length = len(self.tid_list)
        # with lmdb.open(
        #     self.lmdb_features_dir,
        #     map_size=int(self.lmdb_map_size),
        #     readonly=True,
        #     lock=False,
        # ) as lmdb_env:
            # self.length = lmdb_env.stat()["entries"]

    def extract_query(self, queries):
        query_list = set()
        for q in queries:
            for w in q:
                if w in self.noun_set:
                    query_list.add(w)

        query_list = [_ for _ in query_list]
        return query_list
        
    def sample_negative(self, excl_set):
        remains = np.array([_ for _ in self.noun_set if not _ in excl_set])
        idx = [_ for _ in np.random.randint(0, len(remains), self.neg_num_vocab)]
        res = remains[idx]
        return np.array(res) # neg_num_vocab
    

    def sample_negative_views(self, excl_set):
        feats_res = []
        with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
            while len(feats_res) < self.neg_num_views:
                while True: # find a negative path
                    tid = self.tid_list[self.data_idx]
                    vocab_set = self.extract_query(self.instruction_queries[tid])
                    self.data_idx = (self.data_idx + 1) % self.length
                    vocab_set = set(vocab_set)
                    # un = vocab_set.union(excl_set)
                    xset = vocab_set.intersection(excl_set)
                    # print('exe set', excl_set, 'vocab_set', vocab_set,'xset',xset)
            
                    if len(xset) == 0: # if it is a negative path
                        key = tid
                        obs = msgpack_numpy.unpackb(
                                txn.get(str(key).encode()),
                                raw=False,
                            ) # traj_obs
                        feats = obs['rgb_features'] # T x num_panos x D x W x H
                        feats = np.transpose(feats, (0, 1, 3, 4, 2))
                        T, N, W, H, D = feats.shape
                        feats = feats.reshape((-1, D))
                        for feat in feats:
                            if len(feats_res) < self.neg_num_views:
                                feats_res.append(feat)
                            else:
                                break

                        break
                    
        return np.array(feats_res) # neg_num_views x D

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            # lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for _ in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break
                    idx = self.load_ordering.pop()
                    tid = self.tid_list[idx]
                    key = tid
                    item = {}
                    obs = msgpack_numpy.unpackb(
                            txn.get(key.encode()),
                            raw=False,
                        )# traj_obs

                    item['rgb_features'] = obs['rgb_features']
                    query_list = self.extract_query(self.instruction_queries[tid])
                    
                    item['queries'] = np.array(query_list)
                    item['neg_vocabs'] = self.sample_negative(query_list)
                    
                    item['neg_views_feats'] = self.sample_negative_views(query_list)
                    
                    new_preload.append(item)

                    # lengths.append(len(new_preload[-1][0]))

            # sort_priority = list(range(len(lengths)))
            # random.shuffle(sort_priority)

            # sorted_ordering = list(range(len(lengths)))
            # sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            # for idx in _block_shuffle(sorted_ordering, self.batch_size):
            #     self._preload.append(new_preload[idx])
            for item in new_preload:
                self._preload.append(item)

        return self._preload.pop()

    def __next__(self):
        obs = self._load_next()
        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        return obs

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = 0
            end = self.length
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))

            start = per_worker * worker_info.id
            end = min(start + per_worker, self.length)
            self.data_idx = start
        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(start, end)), self.preload_size)
            )
        )

        return self


@baseline_registry.register_trainer(name="collector")
class DataCollecter(BaseVLNCETrainer):
    def __init__(self, config=None):
        self.lmdb_features_dir = config.IL.DAGGER.lmdb_features_dir.format(
            split=config.TASK_CONFIG.DATASET.SPLIT
        )
        config = add_pano_sensors_to_config(config)
        super().__init__(config)

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        os.makedirs(self.lmdb_features_dir, exist_ok=True)
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def _initialize_policy(
        self,
        config: Config,
    ) -> None:
        self.policy = CLEncAgent(config.MODEL)
        self.policy.to(self.device)


    def _update_dataset(self):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        expert_uuid = self.config.IL.DAGGER.expert_policy_sensor_uuid
        
        prev_actions = torch.zeros(
            envs.num_envs,
            1,
            device=self.device,
            dtype=torch.long,
        )
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        observations = envs.reset()
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        self.obs_transforms = get_active_obs_transforms(self.config)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        episodes = [[] for _ in range(envs.num_envs)]
        skips = [False for _ in range(envs.num_envs)]
        # Populate dones with False initially
        dones = [False for _ in range(envs.num_envs)]

        # https://arxiv.org/pdf/1011.0686.pdf
        # Theoretically, any beta function is fine so long as it converges to
        # zero as data_it -> inf. The paper suggests starting with beta = 1 and
        # exponential decay.
        # p = self.config.IL.DAGGER.p
        # # in Python 0.0 ** 0.0 == 1.0, but we want 0.0
        # beta = 0.0 if p == 0.0 else p ** data_it

        # ensure_unique_episodes = beta == 1.0

        ensure_unique_episodes = True


        collected_eps = 0
        ep_ids_collected = None
        if ensure_unique_episodes:
            ep_ids_collected = {
                ep.episode_id for ep in envs.current_episodes()
            }

        with tqdm.tqdm(
            total=self.config.IL.DAGGER.update_size, dynamic_ncols=True
        ) as pbar, lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.config.IL.DAGGER.lmdb_map_size),
        ) as lmdb_env, torch.no_grad():
            start_id = lmdb_env.stat()["entries"]
            txn = lmdb_env.begin(write=True)

            # while collected_eps < self.config.IL.DAGGER.update_size:
            while envs.num_envs > 0:
                current_episodes = None
                envs_to_pause = None
                if ensure_unique_episodes:
                    envs_to_pause = []
                    current_episodes = envs.current_episodes()

                for i in range(envs.num_envs):
                    if dones[i] and not skips[i]:
                        ep_id = current_episodes[i].episode_id
                        traj_id = current_episodes[i].trajectory_id
                        ep = episodes[i]
                        traj_obs = batch_obs(
                            [step[0] for step in ep],
                            device=torch.device("cpu"),
                        )
                        del traj_obs[expert_uuid]
                        traj_obs_t = {}
                        for k, v in traj_obs.items():
                            if not k in ['instruction', 'rgb_features']:
                                # del traj_obs[k]
                                continue
                            # print(k, v.shape)
                            traj_obs_t[k] = v.numpy()
                            if self.config.IL.DAGGER.lmdb_fp16:
                                traj_obs_t[k] = traj_obs_t[k].astype(np.float16)
                        traj_obs = traj_obs_t
                        transposed_ep = [
                            traj_obs,
                            np.array([step[1] for step in ep], dtype=np.int64),
                            np.array([step[2] for step in ep], dtype=np.int64),
                        ]
                        txn.put(
                            f'{ep_id}_{traj_id}'.encode(),
                            msgpack_numpy.packb(
                                transposed_ep, use_bin_type=True
                            ),
                        )

                        pbar.update()
                        collected_eps += 1

                        if (
                            collected_eps
                            % self.config.IL.DAGGER.lmdb_commit_frequency
                        ) == 0:
                            txn.commit()
                            txn = lmdb_env.begin(write=True)

                        if ensure_unique_episodes:
                            if (
                                current_episodes[i].episode_id
                                in ep_ids_collected
                            ):
                                envs_to_pause.append(i)
                            else:
                                ep_ids_collected.add(
                                    current_episodes[i].episode_id
                                )

                    if dones[i]:
                        episodes[i] = []

                if ensure_unique_episodes:
                    (
                        envs,
                        prev_actions,
                        batch,
                        _,
                    ) = self._pause_envs(
                        envs_to_pause,
                        envs,
                        prev_actions,
                        batch,
                    )
                    if envs.num_envs == 0:
                        break
                
                # for key, v in batch.items():
                #     print(key, v.shape)

                rgb_features = self.policy.forward(batch)
                
                actions = batch[expert_uuid].long()

                
                

                for i in range(envs.num_envs):
                    if rgb_features is not None:
                        observations[i]["rgb_features"] = rgb_features[i].cpu().numpy()
                        del observations[i]["rgb"]

                    episodes[i].append(
                        (
                            observations[i],
                            prev_actions[i].item(),
                            batch[expert_uuid][i].item(),
                        )
                    )

                skips = batch[expert_uuid].long() == -1
                actions = torch.where(
                    skips, torch.zeros_like(actions), actions
                )
                skips = skips.squeeze(-1).to(device="cpu", non_blocking=True)
                prev_actions.copy_(actions)
                exe_actions = [a[0].item() for a in actions]
                # print('actions', exe_actions)

                outputs = envs.step(exe_actions)
                observations, _, dones, _ = [list(x) for x in zip(*outputs)]
                observations = extract_instruction_tokens(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=self.device,
                )

            txn.commit()

        envs.close()
        envs = None

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        prev_actions,
        batch,
        rgb_frames=None,
    ):
        # pausing envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            prev_actions,
            batch,
            rgb_frames,
        )


    def train(self) -> None:
        """Main method for training DAgger."""
        if self.config.IL.DAGGER.preload_lmdb_features:
            try:
                lmdb.open(self.lmdb_features_dir, readonly=True)
            except lmdb.Error as err:
                logger.error(
                    "Cannot open database for teacher forcing preload."
                )
                raise err
        else:
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.config.IL.DAGGER.lmdb_map_size),
            ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                txn.drop(lmdb_env.open_db())

        EPS = self.config.IL.DAGGER.expert_policy_sensor
        if EPS not in self.config.TASK_CONFIG.TASK.SENSORS:
            self.config.TASK_CONFIG.TASK.SENSORS.append(EPS)

        self.config.defrost()

        # if doing teacher forcing, don't switch the scene until it is complete
        if self.config.IL.DAGGER.p == 1.0:
            self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
                -1
            )
        self.config.freeze()

        self._initialize_policy(
            self.config
        )

        self._update_dataset()

        

            
@baseline_registry.register_trainer(name="embedding_trainer")
class EmbeddingTrainer(BaseTrainer):
    def __init__(self, config=None):
        self.config = config
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        # self.lmdb_features_dir = 'data/trajectories_dirs/pano_gt/trajectories.lmdb'
        # self.lmdb_features_dir = 'data/trajectories_dirs/pano_gt/feats.lmdb'
        self.instructions_path = 'data/trajectories_dirs/pano_gt/instructions.pkl'
        self.key_path = 'data/trajectories_dirs/pano_gt/keys.pkl'
        self.lmdb_features_dir = '../pano_gt/feats.lmdb'
        # self.instructions_path = '../pano_gt/instructions.pkl'
        # super().__init__(config)

    def _initialize_policy(
        self,
        config: Config,
    ) -> None:
        self.policy = CLEncAgent(config.MODEL)
        self.policy.to(self.device)

    def process_data(self):
        self._initialize_policy(self.config)
        lmdb_env = lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.config.IL.DAGGER.lmdb_map_size),
        )
        txn = lmdb_env.begin()
        instruction_queries = {}
        for key, value in txn.cursor():
            v = msgpack_numpy.unpackb(
                            value,
                            raw=False,
                        )
            observation = v[0]
            k = key.decode()
            ep_id, traj_id = k.split('_')
            
            # print(ep_id, traj_id)
            if traj_id in instruction_queries:
                instruction_queries[traj_id].append(observation['instruction'][0])
            else:
                instruction_queries[traj_id] = [observation['instruction'][0]]

            # print(observation['instruction'])
        with open(self.instructions_path,'wb') as fp:
            pickle.dump(instruction_queries, fp)
    

    def sample_negatives(self, excl_set, num):
        remains = np.array([_ for _ in self.noun_set if not _ in excl_set])
        idx = [_ for _ in np.random.randint(0,len(remains),num)]
        res = remains[idx]
        return np.array(res)

    def sample_negatives_views(self, excl_set, num):
        feats_res = []
        while len(feats_res) < num:
            pass


    def train_bk(self):
        lmdb_env = lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.config.IL.DAGGER.lmdb_map_size),
        )
        txn = lmdb_env.begin()

        lmdb_env_write = lmdb.open('../pano_gt/trajectories.lmdb', map_size=int(self.config.IL.DAGGER.lmdb_map_size))
        txn_write = lmdb_env_write.begin(write=True)

        output = ''

        t = time.time()
        
        for i, (k, v) in enumerate(tqdm.tqdm(txn.cursor())):
            txn_write.put(k, v)
            if i == 10:
                break
        t = time.time() - t
        output += f'{t} seconds\n'

        txn_write.commit()
        txn_write = lmdb_env_write.begin()

        t = time.time()
        for k, v in tqdm.tqdm(txn_write.cursor()):
            print(k)
            pass
            
        t = time.time() - t
        output += f'{t} seconds after\n'
        with open("file.txt",'w') as fp:
            fp.write(output)

    def save_dataset(self):
        key_path = 'data/trajectories_dirs/pano_gt/keys.pkl'
        with open(key_path, 'rb') as fp:
            key_set = pickle.load(fp)
            key_list = [_.decode() for _ in key_set]
        
        tid2kid = {}
        tid_list = []
        for key in key_list:
            eid, tid = key.split('_')
            if not tid in tid2kid:
                tid2kid[tid] = key
                tid_list.append(tid)
        
        lmdb_env = lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.config.IL.DAGGER.lmdb_map_size),
        )
        txn = lmdb_env.begin()
        
        lmdb_env_write = lmdb.open('data/trajectories_dirs/pano_gt/feats.lmdb', map_size=int(self.config.IL.DAGGER.lmdb_map_size))
        txn_write = lmdb_env_write.begin(write=True)
        
        for i, tid in enumerate(tid_list):
            key = tid2kid[tid]
    
            value = txn.get(key.encode())
            obs = msgpack_numpy.unpackb(
                            value,
                            raw=False,
                        )[0] # traj_obs
            
            item = {}
            item['rgb_features'] = obs['rgb_features']
            txn_write.put(
                    tid.encode(),
                    msgpack_numpy.packb(
                        item, use_bin_type=True
                    ),
                )

            if i % 100 == 0 and i > 0:
                txn_write.commit()
                txn_write = lmdb_env_write.begin(write=True)

        txn_write.commit()
            
    def trans_obs(self, obs):
        for key, v_list in obs.items():
            for i, v in enumerate(v_list):
                v_list[i] = v.to(self.device)
        
        return obs

    def train(self):
        batch_size = self.config.NUM_ENVIRONMENTS

        dataset = IWTrajectoryDataset(
            self.lmdb_features_dir,
            self.key_path,
            self.instructions_path,
            lmdb_map_size=self.config.IL.DAGGER.lmdb_map_size,
            batch_size=batch_size,
        )

        # item = next(dataset)
        # for key in item:
        #     print(key, item[key].shape)
        
        diter = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,  # drop last batch if smaller
            num_workers=3,
        )

        data_iter = iter(diter)
        # batch = next(data_iter)
        # for key in batch:
        #     print(key, len(batch[key]))
        #     for item in batch[key]:
        #         print(item.shape)
        
        self._initialize_policy(self.config)

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR,
            flush_secs=30,
            purge_step=0,
        ) as writer:
            logs = defaultdict(list)
            for i in range(100000):
                try:
                    batch = next(data_iter)
                    # assert False, 'yes'
                except:
                    data_iter = iter(diter)
                    batch = next(data_iter)

                batch = self.trans_obs(batch)
                
                self.policy.train(batch)

                for key, v in self.policy.logs.items():
                    logs[key].append(v)

                if i % 10 == 0:
                    for key, v in logs.items():
                        value = np.mean(v)
                        writer.add_scalar(f'loss/{key}', value, i)
                        # print(i, key, value)

                    logs = defaultdict(list)

                if i % 100 == 0:
                    self.policy.save(i, os.path.join(self.config.CHECKPOINT_FOLDER, 'LAST_ITER'))
                    
        