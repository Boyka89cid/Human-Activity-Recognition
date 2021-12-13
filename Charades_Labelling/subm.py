import numpy as np
import os
import pandas as pd
from collections import namedtuple, deque, OrderedDict
from operator import attrgetter
import pims


path_str = '/data/ai-bandits/datasets/charades/data'

CHAR_Videos = os.path.join(path_str, "videos")

ClipSample = namedtuple("ClipSample", "dataset videoname start end labels bd_box framerate numframes split")

bdbox = [0,0,1,1]

CHARADES_JSON = "charades_videos.json"


if __name__ == '__main__':

    print(os.getcwd())


    train = pd.read_csv('Charades_v1_train.csv')
    mapfile = pd.read_csv('Charades_v1_mapping.txt',sep = " ", header=None)
    mapfile.columns = ['class','object','verb']
    objfile = pd.read_csv('Charades_v1_objectclasses.txt', sep=" ", header=None)
    objfile.columns = ['object','objval']
    verbfile = pd.read_csv('Charades_v1_verbclasses.txt', sep=" ", header=None)
    verbfile.columns = ['verb','verbval']


    firstjoin = mapfile.merge(objfile, left_on= 'object', right_on='object', how='outer')

    fulldatajoin = firstjoin.merge(verbfile, left_on='verb', right_on='verb', how='outer')


    train_meta_data = train.dropna(subset=['actions']).reset_index(drop=True)

    list_clip_Samples = []

    #print(list(train_meta_data.columns.values))


    with open(CHARADES_JSON, "w") as fp:


        for setofactions, u_id in zip(train_meta_data['actions'], train_meta_data['id']):

            listofactions = setofactions.split(";")


            video_path = os.path.join(CHAR_Videos,"Charades_v1", u_id+".mp4")

            video = pims.Video(video_path)
            #print(0, video.duration, video.frame_rate, len(video))

            st_list = np.array([])
            end_list = np.array([])
            mapping = np.array([])
            ls_tuples = []
            #print(listofactions)


            for action in listofactions:
                Action = namedtuple('Action', ['ClassMap','st','end'])
                meta_action = action.split(" ")
                #print(meta_action)
                onetup = Action(meta_action[0], float(meta_action[1]), float(meta_action[2]))
                ls_tuples.append(onetup)
                #print(onetup)

            #print(ls_tuples)

            ls_tuples_sorted = sorted(ls_tuples, key = attrgetter('st'))

            #print(ls_tuples_sorted)

            # print("\n")
            #
            # print("Each List by index")
            #
            # print(ls_tuples_sorted[0].ClassMap)
            # print(ls_tuples_sorted[1].ClassMap)
            # print (len(ls_tuples_sorted))
            #
            # print("Each list by indexing done")
            # print("\n")

            for eachaction in ls_tuples_sorted:
                mapping = np.append(mapping, eachaction.ClassMap)
                st_list = np.append(st_list, eachaction.st)
                end_list = np.append(end_list, eachaction.end)

            #print (st_list)
            #print (end_list)


            alltimes = deque()

            startls = deque(st_list)
            endls = deque(end_list)

            while startls and endls:
                if startls[0] > endls[0]:
                    alltimes.append(endls.popleft())
                else:
                    alltimes.append(startls.popleft())
            alltimes = alltimes + startls + endls

            alltimes = (list(OrderedDict.fromkeys(list(alltimes))))
            length = len(alltimes)

            #print(alltimes)
            #print(len(alltimes))


            all_classes = {'sit': 0, 'stand': 1, 'lie': 2, 'walk': 3, 'fall': 4, 'eat': 5, 'drink': 6, 'phone': 7, 'paper/notebook': 8,
                           'television': 9, 'cook': 10, 'laptop': 11}


            out = 1
            delta = 0.005

            for i in range(0, length - 1):

                ind_cat = 0
                cls_map_ind = []
                # for st , ed in zip(st_list , end_list):
                #     if st < alltimes[i] + delta and ed > alltimes[i+1] - delta:
                #         cls_map_ind.append(ind_cat)
                #     ind_cat = ind_cat + 1

                cls_map_ind = np.ravel((np.where((st_list < alltimes[i] + delta) & (end_list > alltimes[i+1] - delta))))
                #print(alltimes[i+1])
                #print(cls_map_ind)
                objpass = False
                verbpass = False

                labels = []

                for classind in cls_map_ind:

                    act = ls_tuples_sorted[classind].ClassMap
                    #print(fulldatajoin[fulldatajoin['class'] == act])
                    objcls = fulldatajoin[fulldatajoin['class'] == act]['objval'].values
                    vrbcls = fulldatajoin[fulldatajoin['class'] == act]['verbval'].values

                    try:
                        #print(all_classes[objcls[0]])
                        labels.append(all_classes[objcls[0]])
                        objpass = True
                    except KeyError:
                        pass

                    try:
                        #print(all_classes[vrbcls[0]])
                        labels.append(all_classes[vrbcls[0]])
                        verbpass = True
                    except KeyError:
                        pass

                list_clip_Samples.append(ClipSample('charades', u_id, alltimes[i], alltimes[i+1], labels, bdbox, video.frame_rate, len(video), 1))


            print(list_clip_Samples)
            print(aa)

        print(len(list_clip_Samples))









