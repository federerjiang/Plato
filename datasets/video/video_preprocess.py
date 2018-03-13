import os
import fnmatch
import copy


QP = [10, 15, 20, 25, 30]


def get_info(video):
    bash_cmd = 'ffprobe -v quiet -print_format json -show_format -show_streams ' + video + \
               '> original_video_info.txt'
    os.system(bash_cmd)


def cut_video(video, seg_length):
    bash_cmd = 'ffmpeg -i ' + video + '-c:v libx264 \
    -crf 22 -map 0 -segment_time ' + seg_length + ' -g ' + seg_length + ' -sc_threshold 0 \
    -force_key_frames "expr:gte(t,n_forced*1)" -f segment output%03d.mp4'
    os.system(bash_cmd)


def get_size(video_folder, videos):
    for video in videos:
        video_path = video_folder + video
        if os.path.isfile(video_path):
            file_info = os.stat(video_path)
            size = file_info.st_size / (1024 * 1024)
            # print(str(size) + '   ' + video)
            small_videos = []
            if size < 10.0:
                small_videos.append(video)
            for video in sorted(small_videos):
                print(video)


def prepare_seg_video(video_folder):
    for root, dirnames, filenames in os.walk(os.path.abspath(video_folder)):
        for filename in fnmatch.filter(filenames, '*.mp4'):
            video_name = filename.split('.')[0][6:9]
            print(video_name)
            seg_directory = os.path.join(root, video_name)
            print(seg_directory)
            os.mkdir(seg_directory)
            des_path = os.path.join(seg_directory, filename)
            src_path = os.path.join(root, filename)
            os.rename(src_path, des_path)


def cmd_kvazaar(in_file, out_file, tiles, qp):
    bash_cmd = 'kvazaar -i ' + in_file + ' --input-res 3840x2160 -o ' \
                + out_file + ' --tiles ' + tiles + ' --slices tiles --qp ' \
                + qp
    os.system(bash_cmd)
    print("finished kvazaar command")


def cmd_mp4box(in_path, out_path1, out_path2):
    # 'MP4Box -add 005.hvc:split_tiles -new 005.mp4'
    # 'MP4Box -dash 1000 -rap -frag-rap -profile live -out dash_tiled_011-l6.mpd out_tiled_011-l6.mp4'
    bash_cmd_1 = 'MP4Box -add ' + in_path + ':split_tiles -new ' + out_path1
    bash_cmd_2 = 'MP4Box -dash 1000 -rap -frag-rap -profile live -out ' + out_path2 + ' ' + out_path1
    os.system(bash_cmd_1)
    print("add new video to MP4Box")
    os.system(bash_cmd_2)
    print("get tiled videos")


def prepare_qp_tile_video(video_folder):
    for root, dirnames, filenames in os.walk(os.path.abspath(video_folder)):
        for filename in fnmatch.filter(filenames, '*.mp4'):
            # print(root, filename)
            video_name = filename.split('.')[0][6:9] + '.hvc'
            mp4_name = filename.split('.')[0][6:9] + '.mp4'
            mpd_name = filename.split('.')[0][6:9] + '.mpd'
            in_path = os.path.join(root, filename)
            tiles = '12x6'
            for i in range(len(QP)):
                qp = str(QP[i])
                qp_directory = os.path.join(root, qp)
                os.mkdir(qp_directory)
                out_path = os.path.join(qp_directory, video_name)
                out_path_mp4 = os.path.join(qp_directory, mp4_name)
                out_path_mpd = os.path.join(qp_directory, mpd_name)
                cmd_kvazaar(in_path, out_path, tiles, qp)
                cmd_mp4box(out_path, out_path_mp4, out_path_mpd)


def extract_m4s(src_folder, des_folder):
    for root, dirnames, filenames in os.walk(src_folder):
        for filename in fnmatch.filter(filenames, '*.m4s'):
            file_path = os.path.join(root, filename)
            parse = file_path.split('/')
            m4s_path = parse[-1]
            qp_path = parse[-2]
            seg_path = parse[-3]
            des_folder = os.path.abspath(des_folder)
            seg_path = os.path.join(des_folder, seg_path)
            qp_path = os.path.join(des_folder, seg_path, qp_path)
            if not os.path.isdir(seg_path):
                os.mkdir(seg_path)
            if not os.path.isdir(qp_path):
                os.mkdir(qp_path)
            des_path = os.path.join(qp_path, m4s_path)
            copy_m4s = copy.deepcopy(file_path)
            src_path = os.path.abspath(copy_m4s)
            os.rename(src_path, des_path)


if __name__ == '__main__':
    video_folder = '../tile-sJxiPiAaB4K/'
    print(os.listdir(video_folder))
    video = 'sJxiPiAaB4k.mkv'
    video = video_folder + video
    # get_info(video)
    # cut_video(video=video, seg_length=1)
    # get_size(video_folder, os.listdir(video_folder))
    # prepare_seg_video(video_folder)
    # prepare_qp_tile_video(video_folder)
    extract_m4s(video_folder, '../video_paris_tiles_12x6/')



