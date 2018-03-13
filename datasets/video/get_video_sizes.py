import os

video_path = '/Users/federerjiang/research-project/vr-video-streaming-Sigcomm2018VRAR/video_paris_tiles_12x6'

QP = [10, 15, 20, 25, 30]
seg_total = 237
tile_total = 72

count = 0
video_sizes = {10: [x[:] for x in [[0] * tile_total] * seg_total],
               15: [x[:] for x in [[0] * tile_total] * seg_total],
               20: [x[:] for x in [[0] * tile_total] * seg_total],
               25: [x[:] for x in [[0] * tile_total] * seg_total],
               30: [x[:] for x in [[0] * tile_total] * seg_total]}

for root, dirnames, filenames in os.walk(video_path):
    for filename in filenames:
        parse = root.split('/')
        tile_num = filename[14:-6]
        tile_num = int(tile_num) - 1
        qp = int(parse[-1])
        seg_num = parse[-2]
        seg_num = int(seg_num) - 4
        tile_size = os.path.getsize(os.path.join(root, filename))
        if tile_num == 0:
            count += 1
            continue
        video_sizes[qp][seg_num-1][tile_num-1] = tile_size
        # print(seg_num, qp, tile_num, tile_size)
        # count += 1
        # print(filenames)

print(count)
# map_qp = {}
for i in range(len(QP)):
    for seg_num in range(seg_total):
            with open('video_size_' + str(i) + '_' + str(seg_num), 'w') as f:
                for tile_num in range(tile_total):
                    tile_size = video_sizes[QP[i]][seg_num][tile_num]
                    f.write(str(tile_size) + '\n')


# def test_get_video_size():
#         video_size = {}
#         for bitrate in range(len(QP)):
#             video_size[bitrate] = {}
#             for seg in range(seg_total):
#                 video_size[bitrate][seg] = []
#                 # each file contains size information of tiles in a segment
#                 with open('video_size_' + str(bitrate) + '_' + str(seg)) as f:
#                     for line in f:
#                         video_size[bitrate][seg].append(int(line.split()[0]))
#
#         return video_size
#
# if __name__ == '__main__':
#     video_sizes = test_get_video_size()
#     print(video_sizes[0][233][9])
