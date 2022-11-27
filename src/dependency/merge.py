import os
import json
from tqdm import tqdm
import random


def change_tag(change_list, tag_list):
    new_tag_list = []
    for tag_li_idx, tag_li in enumerate(tag_list):
        new_tag_li = []
        for change in change_list:
            change_result = set(sorted(change[1]))
            for word_idx in change[0]:
                for tag_idx, tag in enumerate(tag_li):
                    ## tag = [[피지배소idx, 지배소idx], [구문정보, 기능정보]]
                    if list(tag[0])[0] == word_idx:
                        tag_list[tag_li_idx][tag_idx] = [[change_result, set(sorted(list(tag[0])[1]))], tag[1]]
                    if list(tag[0])[1] == word_idx:
                        tag_list[tag_li_idx][tag_idx] = [[set(sorted(list(tag[0])[0])), change_result], tag[1]]
        for tag in tag_li:
            if (len(tag[0][0].difference(tag[0][1]))!=0):
                if (len(tag[0][1].difference(tag[0][0]))!=0):new_tag_li.append(tag)
        new_tag_list.append(new_tag_li)
    tag_list = new_tag_list
    del new_tag_list
    return tag_list

def tag_case1(change_list, tag_l):
    ## tag = [[피지배소idx, 지배소idx], [구문정보, 기능정보]]
    case1_conti = True;cnt = 0
    while case1_conti:
        case1_conti = False
        del_tag_l = []
        for tag1 in tag_l:
            for tag2 in tag_l:
                if (max(tag1[0][1]) == min(tag2[0][0])):
                    case1_conti = True;cnt += 1
                    change = tag1[0][0].union(tag1[0][1].union(tag2[0][0]))
                    change_list.append([[tag1[0][0], tag1[0][1], tag2[0][0]], change])
                    tag2[0][0] = change
                    if tag1 in tag_l: del_tag_l.append(tag1)
        new_tag_l = [tag for tag in tag_l if tag not in del_tag_l]
        tag_l = new_tag_l
        del new_tag_l
    return change_list, tag_l, cnt

def tag_case2(change_list, tag_li1, tag_li2):
    ## tag = [[피지배소idx, 지배소idx], [구문정보, 기능정보]]
    case2_conti = True;cnt = 0
    while case2_conti:
        case2_conti = False
        for tag1 in tag_li1:
            del_tag_li2 = []
            for tag2 in tag_li2:
                if ((tag1[0][1] == tag2[0][1]) and ((max(tag1[0][0]) - min(tag2[0][0])) == 1)):
                    case2_conti = True;cnt+=1
                    change = tag1[0][0].union(tag2[0][0])
                    change_list.append([[tag1[0][0], tag2[0][0]], change])
                    tag1[0][0] = change
                    if tag2 in tag_li2: del_tag_li2.append(tag2)
            new_tag_li2 = [tag for tag in tag_li2 if tag not in del_tag_li2]
            tag_li2 = new_tag_li2
            del new_tag_li2

    return change_list, tag_li1, tag_li2, cnt

def merge_tag(datas, CNJ=True):
    outputs = []
    for id, data in tqdm(enumerate(datas)):
        # [{'R', 'VNP', 'L', 'VP', 'S', 'AP', 'NP', 'DP', 'IP', 'X'}, {'None', 'MOD', 'CNJ', 'AJT', 'OBJ', 'SBJ', 'CMP'}]
        # 구문 정보
        r_list = [];
        l_list = [];
        s_list = [];
        x_list = [];
        np_list = [];
        dp_list = [];
        vp_list = [];
        vnp_list = [];
        ap_list = [];
        ip_list = []
        # 수식어 기능 정보
        tag_list = [];
        np_cnj_list = []

        # sentence word
        sen_words = data["preprocess"].split()
        # 지배소 idx
        heads = [x-1 for x in data["parsing"]["heads"][:-1]]

        # 의존관계태그
        labels = data["parsing"]["label"][:-1]
        assert len(sen_words)-1 == len(heads) == len(labels)

        # 문장내 의존관계태그 분류
        for w,(w1, w2_idx, label) in enumerate(zip(sen_words, heads, labels)):
            label = label.split("_")
            if (len(label)==1):label.append("None")

            dependency_list = [[set([w]),set([w2_idx])], label]

            if (label[0] == "NP"):
                if CNJ:
                    if (label[1] == "CNJ"):
                        np_cnj_list.append(dependency_list)
                    if (label[1] != "CNJ"):
                        np_list.append(dependency_list)
                else:np_list.append(dependency_list)
            elif (label[0] == "VP"):
                vp_list.append(dependency_list)
            elif (label[0] == "VNP"):
                vnp_list.append(dependency_list)
            elif (label[0] == "DP"):
                dp_list.append(dependency_list)
            elif (label[0] == "AP"):
                ap_list.append(dependency_list)
            elif (label[0] == "IP"):
                ip_list.append(dependency_list)
            elif (label[0] == "R"):
                r_list.append(dependency_list)
            elif (label[0] == "L"):
                l_list.append(dependency_list)
            elif (label[0] == "S"):
                s_list.append(dependency_list)
            elif (label[0] == "X"):
                x_list.append(dependency_list)

            if (label[1] in ["MOD", "AJT", "CMP", "None"]):
                tag_list.append(dependency_list);
        vp_list = vp_list + vnp_list

        tag_list = [tag_list] + [x for x in [np_list, dp_list, vp_list, ap_list, ip_list, r_list, l_list, s_list, x_list] if len(x) != 0]

        # NP-CNJ
        if np_cnj_list != []:
            np_cnj_list = [cnj[0] for cnj in np_cnj_list]
            for word_idxs in np_cnj_list:
                for tag_li_idx,tag_li in enumerate(tag_list):
                    new_tag_li = []; new_tag_li2 = []
                    for tag_idx, tag in enumerate(tag_li):
                        if (list(tag[0])[0] == word_idxs[0]):
                            if (word_idxs[0] != list(tag[0])[1]): new_tag_li.append([[word_idxs[0],list(tag[0])[1]], tag[1]])
                        elif (list(tag[0])[1] == word_idxs[0]):
                            if (list(tag[0])[0] != word_idxs[0]): new_tag_li.append([[list(tag[0])[0],word_idxs[0]], tag[1]])
                        elif (list(tag[0])[0] == word_idxs[1]):
                            if (word_idxs[1] != list(tag[0])[1]): new_tag_li.append([[word_idxs[1],list(tag[0])[1]], tag[1]])
                        elif (list(tag[0])[1] == word_idxs[1]):
                            if (list(tag[0])[0] != word_idxs[1]): new_tag_li.append([[list(tag[0])[0],word_idxs[1]], tag[1]])
                    for new_tag in new_tag_li:
                        if new_tag not in tag_li:
                            new_tag_li2.append(new_tag)
                    del new_tag_li
                    new_tag_li2 = new_tag_li2 + tag_li
                    tag_list[tag_li_idx] = new_tag_li2

        #print("len of dependency_list:"+str(len(sum(tag_list, []))))
        Done = True
        while Done:
            origin_tag_list = tag_list.copy()
            for tag_li_idx, tag_li in enumerate(tag_list):
                ## tag_li = [tag1, tag2, ...]
                ## tag = [[set([피지배소idx]), set([지배소idx])], [구문정보, 기능정보]]
                conti = True
                while conti:
                    conti_tf = 0

                    ## case1
                    ## (a<-b<-c)  => (ab<-c)
                    change_list = []
                    tag_dist_1 = []# 지배소와 피지배소 물리적 거리가 1
                    for tag in tag_li:
                        if ((max(list(tag[0])[1])-min(list(tag[0])[0])) == 1):tag_dist_1.append(tag)

                    change_list, tag_dist_1, cnt = tag_case1(change_list, tag_dist_1)
                    if (cnt != 0):
                        conti_tf += cnt
                        tag_list = change_tag(change_list, tag_list)
                        tag_li = tag_list[tag_li_idx]
                        #print("tag_case1 done")

                    ## case2
                    ## (a<-b, a<-c)  => (ab<-c)
                    change_list = []
                    tag_dist_1 = []  # 지배소와 피지배소 물리적 거리가 1
                    tag_dist_2 = []  # 지배소와 피지배소 물리적 거리가 2

                    for tag in tag_li:
                        if ((max(list(tag[0])[1])-min(list(tag[0])[0])) == 1):tag_dist_1.append(tag)
                        elif((max(list(tag[0])[1])-min(list(tag[0])[0])) == 2):tag_dist_2.append(tag)


                    change_list, tag_dist_1, tag_dist_2, cnt = tag_case2(change_list, tag_dist_1, tag_dist_2)

                    if (cnt != 0):
                        conti_tf += cnt
                        tag_list = change_tag(change_list, tag_list)
                        tag_li = tag_list[tag_li_idx]
                        #print("tag_case2 done")

                    if conti_tf == 0: conti = False

            if (origin_tag_list == tag_list): Done = False

        dependency_lists = sum(tag_list, [])

        sen_idxs = [set(), set()]
        for dep_idx, dependency_list in enumerate(dependency_lists):
            # dependency_list = [[set([w]), set([w2_idx])], label]
            word_idxs = dependency_list[0]
            sen_idxs[0].add(min(word_idxs[0]))
            sen_idxs[0].add(min(word_idxs[1]))
            sen_idxs[1].add(max(word_idxs[0])+1)
            sen_idxs[1].add(max(word_idxs[1])+1)

        # 후처리
        ## 삭제
        sen_idxs[0] = set(list(sen_idxs[0])[1:]); sen_idxs[1] = set(list(sen_idxs[1])[:-1])
        if len(sen_idxs[0]) != len(sen_idxs[1]):
            # print(sen_idxs[0].difference(sen_idxs[1]))
            # print(sen_idxs[1].difference(sen_idxs[0]))
            # print("len of dependency_lists: "+str(len(dependency_lists)))
            del_dependency_lists = []
            for dep_idx, dependency_list in enumerate(dependency_lists):
                # dependency_list = [[set([w]), set([w2_idx])], label]
                if min(dependency_list[0][0]) in sen_idxs[0].difference(sen_idxs[1]):
                    if dependency_list in dependency_lists: del_dependency_lists.append(dependency_list)
                if min(dependency_list[0][1]) in sen_idxs[0].difference(sen_idxs[1]):
                    if dependency_list in dependency_lists: del_dependency_lists.append(dependency_list)
                if (max(dependency_list[0][0])+1) in sen_idxs[1].difference(sen_idxs[0]):
                    if dependency_list in dependency_lists: del_dependency_lists.append(dependency_list)
                if (max(dependency_list[0][1])+1) in sen_idxs[1].difference(sen_idxs[0]):
                    if dependency_list in dependency_lists: del_dependency_lists.append(dependency_list)
            new_dependency_lists = [dependency_list for dependency_list in dependency_lists if dependency_list not in del_dependency_lists]
            dependency_lists = new_dependency_lists
            del new_dependency_lists
        #print("len of dependency_lists: " + str(len(dependency_lists)))

        sen_idxs = [set(), set()]
        for dep_idx, dependency_list in enumerate(dependency_lists):
            # dependency_list = [[set([w]), set([w2_idx])], label]
            word_idxs = dependency_list[0]
            sen_idxs[0].add(min(word_idxs[0]))
            sen_idxs[0].add(min(word_idxs[1]))
            sen_idxs[1].add(max(word_idxs[0]) + 1)
            sen_idxs[1].add(max(word_idxs[1]) + 1)
            dependency_lists[dep_idx] = [[" ".join(sen_words[min(word_idxs[0]): 1 + max(word_idxs[0])]), " ".join(
                sen_words[min(word_idxs[1]): 1 + max(word_idxs[1])])]] + dependency_list


        new_sen_words = []
        for start_idx, end_idx in zip(sorted(sen_idxs[0]), sorted(sen_idxs[1])):
            new_sen_words.append([" ".join(sen_words[start_idx: end_idx]), [i for i in range(start_idx, end_idx)]])

        for dep_idx, dependency_list in enumerate(dependency_lists):
            dependency_lists[dep_idx][1] = [sorted(dependency_lists[dep_idx][1][0]), sorted(dependency_lists[dep_idx][1][1])]
            # dependency_lists[dep_idx][1] = [new_sen_words.index(dependency_list[0][0]), new_sen_words.index(dependency_list[0][1])]

        output = {"doc_id":data["doc_id"],
                  "sentence": data["sentence"],
                  "preprocess": data["preprocess"],
                  "merge": {
                      "origin": new_sen_words,
                      "parsing": dependency_lists
                    },
                  "keysentence": data["keysentence"],
                  "tag": data["tag"],
                  "coarse_tag": data["coarse_tag"]
                  }
        outputs.append(output)
    return outputs

if __name__ == '__main__':
    inf_dir = "../../data/origin/DP_origin_preprocess.json"
    outf_dir = "../../data/origin/merge_origin_preprocess/origin.json"
    #
    with open(inf_dir, "r", encoding="utf-8") as inf:
        datas = json.load(inf)
    outputs = merge_tag(datas, CNJ=True)
    
    with open(outf_dir, "w", encoding="utf-8") as outf:
        json.dump(outputs, outf, ensure_ascii=False, indent=4)
    outf.close()
    
