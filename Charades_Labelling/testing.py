import json 
import pims

# i = 0
# for line in open('charades_videos.json', 'r'):
# 	d = json.loads(line)['frame_rate']
# 	print(i)

video = pims.Video('9WGMG.mp4')
print(video[0])
