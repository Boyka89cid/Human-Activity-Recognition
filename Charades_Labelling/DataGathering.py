import numpy as np
import os
import pandas as pd
from collections import namedtuple, deque, OrderedDict
from operator import attrgetter
import pims
import json
from constants import box, path_str, delta, prev_lables, prev_uid, datasets, CHAR_Videos

ClipSample = namedtuple("ClipSample", "dataset video start end labels box frame_rate n_frames split")

Action = namedtuple('Action', ['ClassMap', 'st', 'end'])

CHARADES_JSON = "charades_videos_all_activities.json"

list_clip_Samples = []

all_classes={}

def possible_intervals_mergelists(str_list, end_list):
    """
    Method Returns sorted list of merged start and end
    times.
    Example - str_list = [0, 12.6] , end_list = [11.9, 21.2]
    Method returns [0, 11.9, 12.6, 21.2]
    so that clips fragments are now 0 to 11.9, 11.9 to 12.6 so on..

    """

    allpossibletimes = deque()

    str_ls_dq = deque(str_list)
    end_ls_dq = deque(sorted(end_list))

    while str_ls_dq and end_ls_dq:
        if str_ls_dq[0] > end_ls_dq[0]:
            allpossibletimes.append(end_ls_dq.popleft())
        else:
            allpossibletimes.append(str_ls_dq.popleft())
    allpossibletimes = allpossibletimes + str_ls_dq + end_ls_dq

    return list(OrderedDict.fromkeys(list(allpossibletimes)))


def metadata_action_tuple(action):
    """
    Method Returns ActionTuple
    Example - action = c001 0 12.6
    Method returns Action('ClassMap' = c001, 'st' = 0 , 'end' = 12.6)

    """
    meta_action = action.split(" ")
    class_mapping = meta_action[0]
    clip_start = float(meta_action[1])
    clip_end = float(meta_action[2])

    return Action(class_mapping, clip_start, clip_end)


def datajoining():
    """
    Method Returns outerjoin of 3 Mapping files i.e.
    Mapping.txt, objectclasses.txt and verbclasses.txt

    """
    mapfile = pd.read_csv('Charades_v1_mapping.txt', sep=" ", header=None)
    mapfile.columns = ['class', 'object', 'verb']
    objfile = pd.read_csv('Charades_v1_objectclasses.txt', sep=" ", header=None)
    objfile.columns = ['object', 'objval']
    verbfile = pd.read_csv('Charades_v1_verbclasses.txt', sep=" ", header=None)
    verbfile.columns = ['verb', 'verbval']

    firstjoin = mapfile.merge(objfile, left_on='object', right_on='object', how='outer')

    return firstjoin.merge(verbfile, left_on='verb', right_on='verb', how='outer')


def action_ids(data):
    """
    Method Returns actions and ids of test or train

    """
    cleaned_dataset = datareading(data)
    return cleaned_dataset['actions'], cleaned_dataset['id']


def datareading(data):
    """
    Method Returns test or train file after reading csv and dropping invalid records

    """
    dataset = pd.read_csv('Charades_v1_' + data + '.csv')
    return dataset.dropna(subset=['actions']).reset_index(drop=True)


def tuples_sorted_bystart_time(setofactions):
    """
    Method Returns ActionTuple sorted by start time
    Example - actions = [c001 2.0 12.6, c002 0 11.6]
    Method returns [Action('ClassMap' = c002, 'st' = 0 , 'end' = 11.6)]

    """
    ls_tuples = []
    ls_tuples.extend([metadata_action_tuple(action) for action in setofactions.split(";")])
    return sorted(ls_tuples, key=attrgetter('st'))


def video_meta_metrics(uid):
    """
    Method Returns frame_rate and number of frames of video with u_id

    """
    video_path = os.path.join(CHAR_Videos, "Charades_v1", u_id + ".mp4")
    video = pims.Video(video_path)
    num_frames_path = os.path.join(path_str, "images", "Charades_v1_rgb", u_id)
    num_frames = sum(os.path.isfile(os.path.join(num_frames_path, f)) for f in os.listdir(num_frames_path))

    return video.frame_rate, num_frames


def get_objectclass_verbclass(fulldatajoin, ls_tuples_sorted, cls_map_ind):
    """
    Method Returns object or Verb Class exist in current Clip fragment.

    """
    labels = []

    for classind in cls_map_ind:
        act = ls_tuples_sorted[classind].ClassMap
        objcls = fulldatajoin[fulldatajoin['class'] == act]['objval'].values
        vrbcls = fulldatajoin[fulldatajoin['class'] == act]['verbval'].values

        # print(objcls[0])
        # print(vrbcls[0])

        all_classes[objcls[0]] = all_classes.get(objcls[0], len(list(all_classes.keys())))
        all_classes[vrbcls[0]] = all_classes.get(vrbcls[0], len(list(all_classes.keys())))

        try:
            labels.append(all_classes[objcls[0]])
        except KeyError:
            pass

        try:
            labels.append(all_classes[vrbcls[0]])
        except KeyError:
            pass

    return list(OrderedDict.fromkeys(sorted(list(labels))))


def get_labels_interval(st_list, end_list, a, b, fulldatajoin, ls_tuples_sorted):
    cls_map_ind = np.ravel((np.where((np.array(st_list) < a + delta) & (np.array(end_list) > b - delta))))
    return get_objectclass_verbclass(fulldatajoin, ls_tuples_sorted, cls_map_ind)


def overlapping(Samples, start):
    """
    Method Returns bool whether previous clip ending time is equal to
    current clip starting time of same video.

    Example [uid = '467EW', Class = c0129, start = 0.0, end = 12.6] and
    [uid = '467EW', Class = c0129, start = 12.6, end = 21.1] returns True

    """
    return Samples[len(Samples) - 1]['end'] == start


def clip_Sample_tup(u_id, start, end, labels, eachdata):
    """
    Method Returns Sample Clip Tuple

    """
    dataset = 'charades'
    frame_rate, len_vid = video_meta_metrics(u_id)

    return ClipSample(dataset, u_id, start, end, labels, box, frame_rate, len_vid, int(eachdata == 'train'))._asdict()


if __name__ == '__main__':

    fulldatajoin = datajoining()

    with open(CHARADES_JSON, "w") as fp:

        for eachdataset in datasets:

            charades_actions, charades_uids = action_ids(eachdataset)

            for setofactions, u_id in zip(charades_actions, charades_uids):

                ls_tuples_sorted = tuples_sorted_bystart_time(setofactions)
                end_list = []
                st_list = []
                st_list.extend([eachaction.st for eachaction in ls_tuples_sorted])
                end_list.extend([eachaction.end for eachaction in ls_tuples_sorted])
                alltimes = possible_intervals_mergelists(st_list, end_list)

                for i in range(0, len(alltimes) - 1):

                    unique_labels = get_labels_interval(st_list, end_list, alltimes[i], alltimes[i + 1], fulldatajoin,
                                                        ls_tuples_sorted)
                    if prev_uid == u_id and set(prev_lables) == set(unique_labels) and overlapping(list_clip_Samples,
                                                                                                   alltimes[i]):
                        list_clip_Samples[len(list_clip_Samples) - 1]['end'] = alltimes[i + 1]
                    else:
                        list_clip_Samples.append(
                            clip_Sample_tup(u_id, alltimes[i], alltimes[i + 1], unique_labels, eachdataset))

                    prev_lables = unique_labels
                    prev_uid = u_id

                # print(len(list_clip_Samples))
                # print(list_clip_Samples)
                if len(list(all_classes.keys())) == 71:
                    print(all_classes)
                    print("STOP")

        for each in list_clip_Samples:
            fp.writelines(json.dumps(each) + '\n')

        print(len(list_clip_Samples))
        print(len(list(all_classes.keys())))
