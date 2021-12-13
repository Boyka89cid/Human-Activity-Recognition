import numpy as np
import os
import pandas as pd
from collections import namedtuple, deque, OrderedDict
from operator import attrgetter
import pims
import json


path_str = '/data/ai-bandits/datasets/charades/data'

CHAR_Videos = os.path.join(path_str, "videos")

ClipSample = namedtuple("ClipSample", "dataset video start end labels box frame_rate n_frames split")
Action = namedtuple('Action', ['ClassMap','st','end'])

box = [0,0,1,1]

CHARADES_JSON = "charades_videos_bckup1.json"




def possible_intervals_mergelists(str_list, end_list):

    alltimes = deque()

    str_ls_dq = deque(str_list)
    end_ls_dq = deque(sorted(end_list))

    while str_ls_dq and end_ls_dq:
        if str_ls_dq[0] > end_ls_dq[0]:
            alltimes.append(end_ls_dq.popleft())
        else:
            alltimes.append(str_ls_dq.popleft())
    alltimes = alltimes + str_ls_dq + end_ls_dq

    return list(OrderedDict.fromkeys(list(alltimes)))



def metadata_action_tuple(action):

    meta_action = action.split(" ")
    class_mapping = meta_action[0]
    clip_start = float(meta_action[1])
    clip_end = float(meta_action[2])

    return Action(class_mapping, clip_start, clip_end)


def datajoining():

    mapfile = pd.read_csv('Charades_v1_mapping.txt',sep = " ", header=None)
    mapfile.columns = ['class','object','verb']
    objfile = pd.read_csv('Charades_v1_objectclasses.txt', sep=" ", header=None)
    objfile.columns = ['object','objval']
    verbfile = pd.read_csv('Charades_v1_verbclasses.txt', sep=" ", header=None)
    verbfile.columns = ['verb','verbval']


    firstjoin = mapfile.merge(objfile, left_on= 'object', right_on='object', how='outer')

    return firstjoin.merge(verbfile, left_on='verb', right_on='verb', how='outer')


def action_ids(data):

    cleaned_dataset = datareading(data)
    return cleaned_dataset['actions'], cleaned_dataset['id']




def datareading(data):

    dataset = pd.read_csv('Charades_v1_'+data+'.csv')
    return dataset.dropna(subset=['actions']).reset_index(drop=True)


if __name__ == '__main__':

    print(os.getcwd())

    fulldatajoin = datajoining()

    list_clip_Samples = []


    prev_lables = list(range(0,13))
    prev_uid ='NONE'

    datasets = ['train','test']
    split_val = 1

    with open(CHARADES_JSON, "w") as fp:

        for eachdataset in datasets:

            for setofactions, u_id in zip(action_ids(eachdataset)[0],action_ids(eachdataset)[1]):

                listofactions = setofactions.split(";")


                video_path = os.path.join(CHAR_Videos,"Charades_v1", u_id+".mp4")

                video = pims.Video(video_path)
                num_frames_path = os.path.join(path_str,"images","Charades_v1_rgb",u_id)
                cpt = sum(os.path.isfile(os.path.join(num_frames_path, f)) for f in os.listdir(num_frames_path))

                #print(0, video.duration, video.frame_rate, len(video))

                st_list = np.array([])
                end_list = np.array([])
                mapping = np.array([])
                ls_tuples = []
                #print(listofactions)

                ls_tuples.extend([metadata_action_tuple(action) for action in listofactions])
                #print(ls_tuples)

                ls_tuples_sorted = sorted(ls_tuples, key=attrgetter('st'))
                print(ls_tuples_sorted)


                for eachaction in ls_tuples_sorted:
                    mapping = np.append(mapping, eachaction.ClassMap)
                    st_list = np.append(st_list, eachaction.st)
                    end_list = np.append(end_list, eachaction.end)


                alltimes = possible_intervals_mergelists(st_list, end_list)
                length = len(alltimes)


                all_classes = {'sit': 0, 'stand': 1, 'lie': 2, 'walk': 3, 'fall': 4, 'eat': 5, 'drink': 6, 'phone/camera': 7, 'book': 8,
                               'television': 9, 'cook': 10, 'laptop': 11}


                out = 1
                delta = 0.005

                lis_action_dict = []

                for i in range(0, length - 1):

                    ind_cat = 0
                    cls_map_ind = []



                    cls_map_ind = np.ravel((np.where((st_list < alltimes[i] + delta) & (end_list > alltimes[i+1] - delta))))

                    labels = []
                    #print(cls_map_ind)

                    for classind in cls_map_ind:

                        #print(classind)

                        act = ls_tuples_sorted[classind].ClassMap
                        #print(act)
                        #print(fulldatajoin[fulldatajoin['class'] == act])

                        objcls = fulldatajoin[fulldatajoin['class'] == act]['objval'].values
                        vrbcls = fulldatajoin[fulldatajoin['class'] == act]['verbval'].values

                        try:
                            labels.append(all_classes[objcls[0]])
                        except KeyError:
                            pass

                        try:
                            labels.append(all_classes[vrbcls[0]])
                        except KeyError:
                            pass

                        #print(objcls[0], vrbcls[0])
                    unique_labels = list(OrderedDict.fromkeys(sorted(list(labels))))

                    len_dict = len(list_clip_Samples)


                    #print(alltimes[i], alltimes[i+1])
                    print(unique_labels)
                    #print(u_id)

                    if prev_uid == u_id and set(prev_lables) == set(unique_labels) and list_clip_Samples[len_dict - 1]['end'] == alltimes[i]:

                        list_clip_Samples[len_dict - 1]['end'] = alltimes[i+1]

                    else:
                        list_clip_Samples.append(
                            ClipSample('charades', u_id, alltimes[i], alltimes[i + 1], unique_labels, box, video.frame_rate,
                                       cpt, split_val)._asdict())

                    prev_lables = unique_labels
                    prev_uid = u_id

            split_val = 0

        for each in list_clip_Samples:
            fp.writelines(json.dumps(each) + '\n')

        print(len(list_clip_Samples))









