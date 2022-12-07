from copy import deepcopy
from typing import Any
import os
import numpy as np
from gym import Space, spaces
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.dataset import Dataset, Episode
from habitat.core.simulator import Sensor, SensorTypes, Simulator
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from numpy import ndarray
# import librosa
from skimage.measure import block_reduce

import scipy
from scipy.io import wavfile
from scipy.signal import fftconvolve
import torch
import msgpack_numpy

from scipy.spatial import KDTree

from habitat_extensions.shortest_path_follower import (
    ShortestPathFollowerCompat, ShortestPathFollowerOrientation, ShortestPathFollowerWaypoint
)
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis

from habitat_extensions.task import VLNExtendedEpisode
import lmdb
import pickle

def load_metadata(parent_folder):
    points_file = os.path.join(parent_folder, 'points.txt')
    if "replica" in parent_folder:
        graph_file = os.path.join(parent_folder, 'graph.pkl')
        points_data = np.loadtxt(points_file, delimiter="\t")
        points = list(zip(
            points_data[:, 0],
            points_data[:, 1],
            points_data[:, 3] - 1.5528907,
            # points_data[:, 3],
            -points_data[:, 2])
        )
        points = {v[0]: (v[1],v[2],v[3]) for v in points}
    else:
        graph_file = os.path.join(parent_folder, 'graph.pkl')
        points_data = np.loadtxt(points_file, delimiter="\t")
        points = list(zip(
            points_data[:, 0],
            points_data[:, 1],
            points_data[:, 3] - 1.5,
            # points_data[:, 3],
            -points_data[:, 2])
        )
        points = {v[0]: (v[1],v[2],v[3]) for v in points}

    if not os.path.exists(graph_file):
        raise FileExistsError(graph_file + ' does not exist!')
    else:
        with open(graph_file, 'rb') as fo:
            graph = pickle.load(fo)

    ## filter out the points out of the graph (has no audio data)
    tmp = {}
    for p in graph.nodes:
        tmp[p] = points[p]
    points = tmp

    return points, graph

def fft_convolve_torch(a, b):
    with torch.no_grad():
        n = a.shape[-1] + b.shape[-1] -1
        N = 2 ** (int(np.log2(n))+1)
        A = torch.fft.fft(a, N)
        B = torch.fft.fft(b, N)
        return torch.fft.ifft(A*B)[...,:n].real

## additionally sensors
@registry.register_sensor
class SpectrogramSensor(Sensor):
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self.metadata_dir = os.path.join(config.METADATA_DIR, 'mp3d')
        scenes = [_ for _ in os.listdir(self.metadata_dir)]
        self.points = {}
        self.graph = {}
        self._interp_cache = {}
        self._audiogoal_cache = {}
        self.episode_id = None
        self._episode_step_count = 0
        self.num_cameras = config.PANO_ROTATIONS
        self.pairs = {}
        self._binaural_rir_cache = {}
        self.device = (
            torch.device("cuda", config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = torch.device("cpu")
        # print('device', self.device)

        # with open('data/binaural_filenames.pkl','rb') as fp:
        #     file_name_data = pickle.load(fp)
        #     for scene in file_name_data:
        #         data = file_name_data[scene]
        #         s = set()
        #         for o, f in data:
        #             s.add(f[:-4])
        #         self.pairs[scene] = s
        
        for scene in scenes:
            p, g = load_metadata(os.path.join(self.metadata_dir, scene))
            self.points[scene] = p
            self.graph[scene] = g
        
        self.source_sound_dir = config.SOURCE_SOUND_DIR
        self._construct_kdtree()
        self._source_sound_dict = dict()
        self.binaural_rir_dir = config.BINAURAL_RIR_DIR
        self.lmdb_env = lmdb.open(self.binaural_rir_dir,readonly=True, max_readers=512, lock=False)
        self.txn = self.lmdb_env.begin()
        
        super().__init__(config=config)
        self._load_source_sounds()
        print('audio sensor init finished.')

    @staticmethod
    def position_encoding(position):
        return '{:.2f}_{:.2f}_{:.2f}'.format(*position)


    def _read_binfile(self, path):
        if path in self._audiogoal_cache:
            binaural_rir = self._audiogoal_cache[path]
        else:
            
            res = self.txn.get(path.encode())
            if res is not None:
                binaural_rir = msgpack_numpy.unpackb(res, raw=False)
            # try:
                # _, binaural_rir =   # float32
            # except Exception:
            else:
                binaural_rir = np.zeros((1000, 2)).astype(np.float32)
            
            
            if len(binaural_rir) == 0:
                binaural_rir = np.zeros((1000, 2)).astype(np.float32)
        
        return binaural_rir

    def _load_source_sounds(self):
        # load all mono files at once
        sound_files = os.listdir(self.source_sound_dir)
        for sound_file in sound_files:
            sr, audio_data = wavfile.read(os.path.join(self.source_sound_dir, sound_file))
            assert sr == 44100
            if sr != self.config.RIR_SAMPLING_RATE:
                audio_data = scipy.signal.resample(audio_data, self.config.RIR_SAMPLING_RATE)
            self._source_sound_dict[sound_file] = audio_data

    def _construct_kdtree(self):
        self.tree = {}
        self.points_to_index = {}
        for scene, points in self.points.items():
            self.points_to_index[scene] = {self.position_encoding(v): int(u) for u, v in points.items()}
            data = [v for u, v in points.items()]
            # print(scene, len(data))
            if len(data) == 0:
                continue
            self.tree[scene] = KDTree(data)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "spectrogram"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        spectrogram = self.compute_spectrogram(np.ones((2, self.config.RIR_SAMPLING_RATE)))

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=[self.num_cameras, *spectrogram.shape],
            dtype=np.float32,
        )

    @staticmethod
    def compute_spectrogram(audio_data):
        def compute_stft(signal):
            n_fft = 512
            hop_length = 160
            win_length = 400
            stft = np.array(np.abs(scipy.signal.stft(signal, nfft=n_fft)[2])[...,:-1])
            stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
            return stft

        channel1_magnitude = np.log1p(compute_stft(audio_data[0]))
        channel2_magnitude = np.log1p(compute_stft(audio_data[1]))
        spectrogram = np.stack([channel1_magnitude, channel2_magnitude], axis=-1)

        return spectrogram

    def _position_to_index(self, position, source=None):
        tree = self.tree[self.current_scene_name]
        graph = self.graph[self.current_scene_name]
        if source is not None:
            _, idx = tree.query(position, 5)
            _, sid = tree.query(source, 1)
            sid = self._find_index(tree.data[sid])
            for _id in idx:
                po = tree.data[_id]
                res = self._find_index(po)
                try:
                    nx.shortest_path(graph, source=res, target=sid)
                except:
                    continue

                break
        else:
            _, idx = tree.query(position, 1)
            po = tree.data[idx]
            res = self._find_index(po)
        return res


    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        self.current_scene_name = episode.scene_id.split('/')[-2]
        if episode.episode_id != self.episode_id:
            self.episode_id = episode.episode_id
            self._episode_step_count = 0
            self._audio_index = 0
            self._interp_cache = {}
            self._audiogoal_cache = {}
            self._duration = int(episode.duration)
            sound_id = episode.sound_id.split('/')[1]
            self.current_source_sound = self._source_sound_dict[sound_id]
            self._audio_length = self.current_source_sound.shape[0]// self.config.RIR_SAMPLING_RATE
            self._source_position_index = self._position_to_index(episode.goals[0].position)

            
            
        state = self._sim.get_agent(0).get_state()
        self._receiver_position = state.position
        self._receiver_position_index = self._position_to_index(self._receiver_position)
        rotation = state.rotation
        self._rotation_angle = int(np.around(np.rad2deg(quat_to_angle_axis(rotation)[0]))) % 360


        spectrogram = self.get_current_spectrogram_observation(self.compute_spectrogram)

        self._episode_step_count += 1

        return spectrogram

    

    def get_current_spectrogram_observation(self, audiogoal2spectrogram):
        try:
            # return self._spectrogram_cache[sr_index]
            audiogoal = self._compute_audiogoal()
            
            spectrogram = []
            for audio in audiogoal:
                spect = audiogoal2spectrogram(audio)
                spectrogram.append(spect)
            
            spectrogram = np.stack(spectrogram)
            # self.time_dict['spect'] += time.time() - st
            return spectrogram
        except:
            with open('output_logs/env_error.txt','a') as fp:
                traceback.print_exc(file=fp)
                
            audiogoal = [np.zeros(1000) for _ in range(2)]
            spectrogram = []
            for audio in audiogoal:
                spect = audiogoal2spectrogram(audio)
                spectrogram.append(spect)
            
            spectrogram = np.stack(spectrogram)
            # self.time_dict['spect'] += time.time() - st
            return spectrogram

    def _binaural_interpolation(self, ind, rotation_angle):
        cache_name = f'{int(ind)}_{int(self._source_position_index)}_{rotation_angle}'
        # print(cache_name)
        if cache_name in self._interp_cache:
            return self._interp_cache[cache_name]
        
        bins = [90, 180, 270, 360]
        for angle in bins:
            if rotation_angle < angle:
                # align_left_path = os.path.join(self.alignment_dir, str(angle-90), '{}_{}_left.pkl'.format(
                # int(ind), int(self._source_position_index)))
                # align_right_path = os.path.join(self.alignment_dir, str(angle-90), '{}_{}_right.pkl'.format(
                # int(ind), int(self._source_position_index)))

                # with open(align_left_path,'rb') as fp:
                #     align_left = pickle.load(fp)
                # with open(align_right_path, 'rb') as fp:
                #     align_right = pickle.load(fp)
                
                key_x = os.path.join('mp3d', self.current_scene_name, str((angle - 90) % 360), '{}_{}.wav'.format(
                int(ind), int(self._source_position_index)))
                key_y = os.path.join('mp3d', self.current_scene_name, str(angle % 360), '{}_{}.wav'.format(
                int(ind), int(self._source_position_index)))

                wy = 1.0 * (rotation_angle - angle + 90) / 90
                wx = 1.0 * (angle - rotation_angle) / 90
                
                x_seq = self._read_binfile(key_x)
                y_seq = self._read_binfile(key_y)
                if not x_seq.shape == y_seq.shape:
                    if x_seq.sum() == 0:
                        x_seq = np.zeros_like(y_seq)
                    if y_seq.sum() == 0:
                        y_seq = np.zeros_like(x_seq)

                ip_res = x_seq * wx + y_seq * wy

                self._interp_cache[cache_name] = ip_res
                return ip_res
        assert True, 'nothing returns'
    
    def _find_index(self, pos):
        encoding = self.position_encoding(pos)
        return self.points_to_index[self.current_scene_name][encoding]

    def _closest_indecies(self, position):
        tree = self.tree[self.current_scene_name]
        dis, idx = tree.query(position, 5)
        pos = [tree.data[i] for i in idx]
        node = [self._find_index(p) for p in pos]
        pairs = []
        for n,d in zip(node, dis):
            pair = "{}_{}".format(int(n), int(self._source_position_index))
            # print('pair',pair)
            # if pair in self.pairs[self.current_scene_name]:
            #     pairs.append((n, d))
            pairs.append((n, d))

            if len(pairs) == 4:
                break
        
        pairs = pairs[:1]
        return pairs

    def _compute_audiogoal(self):
        sampling_rate = self.config.RIR_SAMPLING_RATE
        audio_goals = []
        if self._episode_step_count > self._duration:
            # print('out')
            zeros = np.zeros((2, sampling_rate))
            for i in range(self.num_cameras):
                audio_goals.append(zeros)
            return audio_goals
            
        pos = self._receiver_position
        ind_dis = self._closest_indecies(pos)
        indecies = [_[0] for _ in ind_dis] # first 4 indecies
        dis = [1.0 / (1e-10 + _[1]) for _ in ind_dis] # first 4 distances
        total = sum(dis)
        weights = [d / total for d in dis]
        # print('weights',weights,'dis',ind_dis)
        
        index = self._audio_index
        self._audio_index = (self._audio_index + 1) % self._audio_length
        sources = np.zeros((self.num_cameras, 2, sampling_rate))
        bins = np.zeros((self.num_cameras, 2, sampling_rate))
        TURN_ANGLE = 360 // self.num_cameras
        for i in range(self.num_cameras):
            ip_res = np.zeros((1000, 2), dtype=np.float)
            for ind in indecies:
                angle = (self._rotation_angle + i * TURN_ANGLE) % 360
                ip_res = self._binaural_interpolation(ind, angle)
                ip_res = ip_res[:sampling_rate, :]
                cache_name = f'{ind}_{self._receiver_position_index}_{angle}_{index}'
            
            if len(indecies) != 0:
                if cache_name in self._binaural_rir_cache:
                    # audiogoal_dir = self._binaural_rir_cache[cache_name]
                    pass
                else:
                    # by default, convolve in full mode, which preserves the direct sound
                    if self.current_source_sound.shape[0] == sampling_rate:
                        # sound_channel_dim = self.current_source_sound.shape[-1]
                        sources[i, 0, :sampling_rate] = self.current_source_sound[:sampling_rate]
                        sources[i, 1, :sampling_rate] = self.current_source_sound[:sampling_rate]
                        sound_channel_dim = ip_res.shape[0]
                        for channel in range(ip_res.shape[-1]):
                            bins[i, channel, :sound_channel_dim] = ip_res[:, channel]
                        # binaural_convolved = [fftconvolve(self.current_source_sound, ip_res[:, channel]
                        #                         ) for channel in range(ip_res.shape[-1])]

                        # audiogoal_dir = np.array(binaural_convolved)[:, :sampling_rate]
                    else:
                        
                        if index * sampling_rate - ip_res.shape[0] < 0:
                            source_sound = self.current_source_sound[: (index + 1) * sampling_rate][:sampling_rate]
                            sound_channel_dim = source_sound.shape[-1]
                            sources[i, 0, :sound_channel_dim] = source_sound
                            sources[i, 1, :sound_channel_dim] = source_sound
                            sound_channel_dim = ip_res.shape[0]
                            for channel in range(ip_res.shape[-1]):
                                bins[i, channel, :sound_channel_dim] = ip_res[:, channel]
                            # binaural_convolved = np.array([fftconvolve(source_sound, ip_res[:, channel]
                            #                                         ) for channel in range(ip_res.shape[-1])])
                            # audiogoal_dir = binaural_convolved[:, index * sampling_rate: (index + 1) * sampling_rate]
                            # print('idx',index,'length',self._audio_length,'source sound',source_sound.shape, 'convolved', binaural_convolved.shape, 'audiogoal_dir', audiogoal_dir.shape)
                        else:
                            # include reverb from previous time step
                            source_sound = self.current_source_sound[index * sampling_rate - ip_res.shape[0] + 1
                                                                    : (index + 1) * sampling_rate]
                            source_sound = source_sound[: 2 * sampling_rate][:sampling_rate]
                            # binaural_convolved = np.array([fftconvolve(source_sound, ip_res[:, channel], mode='valid',
                            #                                         ) for channel in range(ip_res.shape[-1])])
                            # audiogoal_dir = binaural_convolved
                            sound_channel_dim = source_sound.shape[-1]
                            sources[i, 0, :sound_channel_dim] = source_sound
                            sources[i, 1, :sound_channel_dim] = source_sound
                            sound_channel_dim = ip_res.shape[0]
                            for channel in range(ip_res.shape[-1]):
                                bins[i, channel, :sound_channel_dim] = ip_res[:, channel]


                    # self._binaural_rir_cache[cache_name] = audiogoal_dir
            else: # in case of the point pair does not exist
                # audiogoal_dir = np.zeros([2, self.current_source_sound.shape[0]])
                # audiogoal_dir = np.zeros([2, sampling_rate])
                pass

            # audio_goals.append(audiogoal_dir)
        

        sources = torch.from_numpy(sources).to(self.device)
        bins = torch.from_numpy(bins).to(self.device)
        audio_goals = [audio for audio in fft_convolve_torch(sources, bins)[..., :sampling_rate].detach().cpu().numpy()]
        
        l = self.num_cameras

        del sources
        del bins

        # torch.cuda.empty_cache()
        
        # audio_goals = audio_goals[l//2:] + audio_goals[:l//2]
        # print('audio goal', audiogoal.shape, 'curren sound shape', self.current_source_sound.shape)
        
        # return audiogoal
        return audio_goals
