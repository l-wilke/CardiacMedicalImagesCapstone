import json
import numpy as np
import dicom
import os
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
import random
from helpers_dicom import DicomWrapper as dicomwrapper


def volDict(paths, diff, vtype='orig',STway='sub', ty=None, rls = None, stcheck = None):
    """
    

    Args:
      paths: 
      diff: 
      vtype:  (Default value = 'orig')
      STway:  (Default value = 'sub')
      ty:  (Default value = None)
      rls:  (Default value = None)
      stcheck:  (Default value = None)

    Returns:

    """
    v_dir = '/masvol/output/dsb/volume/'
    v_dict = {}
    no_data = []
#     print (paths)
    for i in range(len(paths)):
        d = paths[i]
        splits = d.split('_')
        if ('176') in d:
            imgsize = 176
        elif ('256') in d:
            imgsize = 256
        v = computeVolume(v_dir+splits[1]+'/'+paths[i]+'/',imgsize, vtype=vtype,STway=STway, ty=ty, rls = rls, stcheck = stcheck)
        print (v_dir+splits[1]+'/'+paths[i]+'/')
        if len(v.keys()) > 0:
            train_df, validate_df, all_df = create_df(v)
            all_no_outliers = removeOutliers(all_df,v, data=None)
#             print (all_no_outliers.columns)
            test = removeLowEstimates(all_no_outliers)
#             print (test.columns)
            all_no_out_noVol = removeNoVolume(test)
            sys, dia, ef = compute_rmse(all_no_out_noVol, 'all')
            v1_sys_low, v1_sys_high,sys_lows,sys_highs = estimateCheck(all_no_out_noVol, actual='Systole',predicted='Systole_P',dev=diff)
            v1_dia_low, v1_dia_high,dia_lows, dia_highs = estimateCheck(all_no_out_noVol, actual='Diastole',predicted='Diastole_P',dev=diff)
            top_sys = getbest(all_no_out_noVol, actual='Systole',predicted='Systole_P',dev=1)
            top_dia = getbest(all_no_out_noVol, actual='Diastole',predicted='Diastole_P',dev=1)
            if d in v_dict:
                v_dict[d[1:]].update({'Vol Method': vtype,
                                   'Systole RMSE': sys,
                                  'Diastole RMSE': dia,
                                  'EF RMSE': ef,
                                  '% Sys Low': v1_sys_low,
                                  '% Sys High': v1_sys_high,
                                  '% Dia Low': v1_dia_low, 
                                  '% Dia High': v1_dia_high, 
                                  'Sys Lows': sys_lows,
                                  'Sys Highs': sys_highs,
                                  'Dia Lows': dia_lows,
                                  'Dia Highs': dia_highs,
                                 'Top Sys': top_sys,
                                 'Top Dia': top_dia})
            else:
                v_dict[d[1:]]=({'Vol Method': vtype,
                             'Systole RMSE': sys,
                                  'Diastole RMSE': dia,
                                  'EF RMSE': ef,
                                  '% Sys Low': v1_sys_low,
                                  '% Sys High': v1_sys_high,
                                  '% Dia Low': v1_dia_low, 
                                  '% Dia High': v1_dia_high,
                                    'Sys Lows': sys_lows,
                                  'Sys Highs': sys_highs,
                                  'Dia Lows': dia_lows,
                                  'Dia Highs': dia_highs,
                           'Top Sys': top_sys,
                           'Top Dia': top_dia})
        else:
            no_data.append(paths[i])
    return (v_dict, no_data)


def computeVolume(path,imgsize, vtype = 'orig', STway = 'sub',ty = None, rls = None, stcheck = None):
    """
    Function that takes the patient dictionary and identifies the volume calculation method that will be performed based on user inputs. 

    Args:
      path: Path of the patient json volume file
      imgsize: size of the image, 176 or 156
      vtype:  (Default value = 'orig') Volume calculation type: orig, zeros, or ST
      STway:  (Default value = 'sub') Use sub or add in order to add the Slice Thickness to the end of the volume calculation
      ty:  (Default value = None) ST enables the user to combine zeros and ST volume calculation
      rls:  (Default value = None) Flag to determine if remove the first or last slice based on area
      stcheck:  (Default value = None) Flag to determine if use the removal of slices based on the gap between slice locations

    Returns:
    Return a dictionary that consists of  each patient's  ESV, EDV, and a dictionary of issues:
    'numSlices': numSlices,'noMinValue': noMin,'noMaxValue': noMax,'noSL': noSL,'minframes': minframes, 'maxframes': maxframes,'age': age,'gender': gender

    """
    Volumes = {}
    for root, dirs, files in os.walk(path):   

        for i in files:
            if os.path.isfile(root+i):
                if ('_'+str(imgsize)+'.') or ('_'+str(imgsize)+'_') in i:
                    split = i.split('_')
                    source = split[0]
                    if ('roi') in i:
                        patient = split[-2]
                    elif ('CR') in i:
                        patient = split[1]
                    else:
                        patient = split[1].strip('.json')
                    patientdict = json.load(open(root + i))
#                     print (patient)
                    if vtype == 'orig':
                        ESV, EDV, issues = getVolume(patient, patientdict,vtype, rls, stcheck)
                    elif vtype == 'ST':
                        ESV, EDV, issues = getVolume(patient, patientdict, vtype, STway, rls, stcheck)
                    elif vtype == 'zeros':
                        ESV, EDV, issues = getVolume(patient, patientdict, vtype, ty, rls, stcheck)

                    Volumes[source+'_'+patient] = {'ESV': ESV, 
                                                  'EDV': EDV,
                                                   'issues': issues}
    return Volumes

def getVolume(patient,patientdict, v_type = 'orig', ty = None, rls = None, stcheck = None):
    """
    Takes the patient dictionary and sorts the slices based on slice location. Removes any set of the frames that are for the same slice. 

    Args:
      patient: Patient Number - string
      patientdict: patient dictionary
      vtype:  (Default value = 'orig') Volume calculation type: orig, zeros, or ST
      ty:  (Default value = None) ST enables the user to combine zeros and ST volume calculation
      rls:  (Default value = None) Flag to determine if remove the first or last slice based on area
      stcheck:  (Default value = None) Flag to determine if use the removal of slices based on the gap between slice locations

    Returns:
      The ESV, EDV, and dictionary of issues for each patient.

    """
    es= {}
    ed = {}
    noSL = []
    noMin = []
    noMax = []
    numSlices = 0
    issues = {}
    minframes = []
    maxframes = []
    sl = 0

    if int(patient) < 500 or int(patient) == 500:
        orig_path = '/masvol/data/dsb/train/'+patient+'/study/'
    elif int(patient) > 500 and int(patient) <= 700:
        orig_path = '/masvol/data/dsb/validate/'+patient+'/study/'
    else:
        orig_path = '/masvol/data/dsb/test/'+patient+'/study/'

    for i in patientdict:
        numSlices = len(patientdict.keys())
        if len(patientdict.keys()) < 5:

                sl +=1

        if patientdict[i]['minSL'] is None: 

            noSL.append(patientdict[i]['zminframe'])
        else: 
            if patientdict[i]['zmin'] ==0:

                noMin.append(patientdict[i]['zminframe'])
            if patientdict[i]['minSL'] in es:

                s_orig = es[patientdict[i]['minSL']]['slice'].split('_')

                lst_idx = minframes.index([es[patientdict[i]['minSL']]['slice'],
                                           patientdict[es[patientdict[i]['minSL']]['slice']]['zminframe'],
                                          patientdict[es[patientdict[i]['minSL']]['slice']]['minSL']])
                s_tmp = i.split('_')

                if int(s_orig[1]) < int(s_tmp[1]):
                    es[patientdict[i]['minSL']].update({'zmin': patientdict[i]['zmin'],
                                          'minST':patientdict[i]['minST'],
                                          'zcounts': patientdict[i]['zcounts'],
                                                       'slice': i})
                    minframes[lst_idx] = [i, patientdict[i]['zminframe'], patientdict[i]['minSL']]
            else:
                es[patientdict[i]['minSL']] = {'zmin': patientdict[i]['zmin'],
                                              'minST':patientdict[i]['minST'],
                                              'zcounts': patientdict[i]['zcounts'],
                                              'slice': i}
                minframes.append([i, patientdict[i]['zminframe'], patientdict[i]['minSL']])
    
        if patientdict[i]['maxSL'] is None:
            noSL.append(patientdict[i]['zmaxframe'])
        else:
            if patientdict[i]['zmax'] ==0:
                noMax.append(patientdict[i]['zmaxframe'])
            if patientdict[i]['maxSL'] in ed:
                s_orig = ed[patientdict[i]['maxSL']]['slice'].split('_')
                lst_idx = maxframes.index([ed[patientdict[i]['maxSL']]['slice'],
                                           patientdict[ed[patientdict[i]['maxSL']]['slice']]['zmaxframe'],
                                          patientdict[ed[patientdict[i]['maxSL']]['slice']]['maxSL']])
                s_tmp = i.split('_')
                if int(s_orig[1]) < int(s_tmp[1]):
                    ed[patientdict[i]['maxSL']].update({'zmax': patientdict[i]['zmax'],
                                                      'maxST':patientdict[i]['maxST'],
                                                      'zcounts': patientdict[i]['zcounts'],
                                                       'slice': i})
                    
                    maxframes[lst_idx] = [i, patientdict[i]['zmaxframe'], patientdict[i]['maxSL']]
            else:
                ed[patientdict[i]['maxSL']] = {'zmax': patientdict[i]['zmax'],
                                          'maxST':patientdict[i]['maxST'],
                                          'zcounts': patientdict[i]['zcounts'],
                                              'slice': i}
                maxframes.append([i,patientdict[i]['zmaxframe'], patientdict[i]['maxSL']])
    a=sorted(es)
    b=sorted(ed)
    a_idx = []
    b_idx = []
    
    for i in range(len(a)-1):
        if np.ceil(abs(a[i] - a[i+1])) == 1:
            a_idx.append(i)
    for i in range(len(a_idx)):
        try:
            a_1 = es[a[a_idx[i]]]['slice'].split('_')
        except:
            a_1 = es[a[a_idx[i-1]]]['slice'].split('_')
        a_2 = es[a[a_idx[i]+1]]['slice'].split('_')
        if int(a_1[1]) < int(a_2[1]):
            try:
                del es[a[a_idx[i]]]
            except:
                pass
        else:
            try:
                del es[a[a_idx[i]+1]]
            except:
                pass
        
    for i in range(len(b)-1):
        if np.ceil(abs(b[i] - b[i+1])) == 1:
            b_idx.append(i)
    for i in range(len(b_idx)):
        try:
            b_1 = ed[b[b_idx[i]]]['slice'].split('_')
        except:
            b_1 = ed[b[b_idx[i-1]]]['slice'].split('_')
        b_2 = ed[b[b_idx[i]+1]]['slice'].split('_')
        if int(b_1[1]) < int(b_2[1]):
            try:
                del ed[b[b_idx[i]]]
            except:
                pass
        else:
            try:
                del ed[b[b_idx[i]+1]]
            except:
                pass            
    if len(maxframes) > 0:
        age_img = maxframes[0][0]+'/'+maxframes[0][1].strip('.npy')
        age = getAge(orig_path,age_img)
        gender = getGender(orig_path,age_img)
#     print (age_img)
    elif len(minframes) > 0:
        age_img = minframes[0][0]+'/'+minframes[0][1].strip('.npy')
        age = getAge(orig_path,age_img)
        gender = getGender(orig_path,age_img)
    else:
        age = None
    issues[patient] = {'numSlices': numSlices,
                        'noMinValue': noMin,
                       'noMaxValue': noMax,
                       'noSL': noSL,
                      'minframes': minframes,
                      'maxframes': maxframes,
                      'age': age,
                      'gender': gender}
    
    if v_type == 'orig':
        ESV, EDV = origVolume(es,ed, rls, stcheck)
        return (ESV, EDV, issues)
    if v_type == 'ST':
        ESV, EDV = STVolume(es,ed,ty, rls, stcheck)
        return(ESV, EDV, issues)
    if v_type == 'zeros':
        ESV, EDV, issues = zerosVolume(es,ed, issues, ty, rls, stcheck)
        return (ESV, EDV, issues)
    
def origVolume(es, ed, rls = None, stcheck = None):
    """
    Calculates the volume of the heart.

    Args:
      es: Dictionary of the ESFs with slice locations as keys and areas as values
      ed: Dictionary for the EDFs with slice locations as keys and areas as values
      rls:  (Default value = None) Flag to determine if to call removal of the last slice based on area
      stcheck:  (Default value = None) Flag to determine if to to remove slices based on the slice locations

    Returns:
      Return the ESV and EDV for the patient record

    """
    ESV = 0
    EDV = 0
    a = sorted(es)
    b = sorted(ed)
    a_idx = []
    b_idx = []
    
    if rls is not None:
        a, b = remove_Last_Slice_Area_Based(es,a,ed,b)

    if stcheck == 'Slice_ST':
        a,b = remove_Slice_ST_Based(a,b, 's')
    
    if stcheck == 'Large_diff':
        a,b = remove_Slice_ST_Based(a,b, 'l')
    
    for i in range(len(a)-1):
        ESVi = (es[a[i]]['zmin'] + es[a[i+1]]['zmin']) * ((abs(a[i]-a[i+1]))/2)
        ESV = ESV + ESVi
    for i in range(len(b)-1):
        EDVi = (ed[b[i]]['zmax'] + ed[b[i+1]]['zmax']) * ((abs(b[i] - b[i+1])/2))
        EDV = EDV + EDVi
    
    ESV = ESV / 1000
    EDV = EDV / 1000
    return (ESV, EDV)

def remove_Last_Slice_Area_Based(es, a, ed, b):
    """
    Removes the first slice if the area is larger than the second slice or removes the last slice if the area is larger than the slice before it. 

    Args:
      es: Dictionary of slice locations and areas for ES
      a: sorted list of ES slice locations
      ed: Dictionary of slice locations and areas for ED
      b: Sorted list of ED slice locations

    Returns:
    Returns a list os slice locations for ES and ED

    """

    if len(a) > 1:
        if es[a[-1]]['zmin'] > es[a[-2]]['zmin']:
            a = a[:-1]
    if len(a) > 1:
        if es[a[0]]['zmin'] > es[a[1]]['zmin']:
            a = a[1:]
    if len(b) > 1:
        if ed[b[-1]]['zmax'] > ed[b[-2]]['zmax']:
            b = b[:-1]
    if len(b) > 1:
        if ed[b[1]]['zmax'] > ed[b[-2]]['zmax']:
            b = b[1:]
        
    return (a,b)
        
def remove_Slice_ST_Based(a,b, TY):
    """
    Removes slices that are either 2*(normal SliceThickness) away from the previous slice or removes slices that are less than the normal SliceThickness away from the previous slice

    Args:
      a: Sorted Slice Locations for ES
      b: Sorted Slice Locations for ED
      TY: Flag to determine sliceSort or sliceRemove

    Returns:

    """
    a_ST = []
    if len(a) > 1:
        for i in range(len(a)-1):
            a_ST.append(np.round(abs(a[i]- a[i+1])))
        try:
            right_ST = np.bincount(a_ST).argmax()
            if right_ST == 1:
                right_ST = 10
        except:
            right_ST = 'Empty a_ST'
        list_try.append([a, a_ST, '*',right_ST, '*', len(a)])
        if TY == 's':
            new_a = sliceSort(a, right_ST)
        if TY == 'l':
            new_a = sliceRemove(a, right_ST)
    
    b_ST = []
    if len(b) > 1:
        for i in range(len(b)-1):
            b_ST.append(np.round(abs(b[i]- b[i+1])))
        try:
            right_ST = np.bincount(b_ST).argmax()
            if right_ST == 1:
                right_ST = 10
        except:
            right_ST = 'Empty a_ST'
#         list_try.append([b, b_ST, '*',right_ST, '*', len(b)])
        if TY == 's':
            new_b = sliceSort(b, right_ST) 
        if TY == 'l':
            new_b = sliceRemove(b, right_ST)

    if len(a) > 1 and len(b) > 1:
        return (new_a,new_b)
    else:
        return (a,b)

def sliceSort(a, rst):
    """
    Function that removes the slice if it's slice thickness is greater than 2*normal Slice Thickness or if the thickness between the slices is less than the normal Slice Thickness

    Args:
      a: sorted slice thicknesses
      rst: normal Slice Thickness

    Returns:
      list of slice locations that should remain in the volume calculation

    """
    lst_del = []
    new_lsts = []
    for i in range(len(a)-1):
        if i == 0 :
            left = a[i]
#         print (left, a[i+1], rst)
        if (np.round(abs(left - a[i+1])) >= rst) and (np.round(abs(left - a[i+1]) <= (2*rst))):
#             print (i,left, a[i+1])
            new_lsts.append(left)
            left = a[i+1]
            if (i+2) == len(a):
                new_lsts.append(left)
            else:
#                 print (a[i+1])
                lst_del.append(i+1)
    list_test.append([new_lsts, lst_del])
    return (new_lsts)

def sliceRemove(a, rst):
    """
     Function that removes the slice if it's slice thickness is greater than 2*normal Slice Thickness 

    Args:
      a: sorted slice thicknesses
      rst: normal Slice Thickness 

    Returns:
      list of slice locations that should remain in the volume calculation

    """
    lst_del = []
    new_lsts = []
    for i in range(len(a)-1):
        if i == 0 :
            left = a[i]
#         print (left, a[i+1], rst)
        if (np.round(abs(left - a[i+1]) <= (2*rst))):
#             print (i,left, a[i+1])
            new_lsts.append(left)
            left = a[i+1]
            if (i+2) == len(a):
                new_lsts.append(left)
            else:
#                 print (a[i+1])
                lst_del.append(i+1)
    list_test.append([new_lsts, lst_del])
    return (new_lsts)

def STVolume(es, ed, STway = 'sub', rls = None, stcheck = None):
    """
    Volume calculation for the heart, same as the orig volume calculation, but add the slice thickeness of the last slice to the volume. 

    Args:
      es: Dictionary of the ESFs with slice locations as keys and areas as values
      ed: Dictionary for the EDFs with slice locations as keys and areas as values
      rls:  (Default value = None) Flag to determine if to call removal of the last slice based on area
      stcheck:  (Default value = None) Flag to determine if to to remove slices based on the slice locations

    Returns:
      Returns the ESV and EDV for the patient

    """
    ESV = 0
    EDV = 0
    
    a = sorted(es)
    b = sorted(ed)
    
    if rls is not None:
        a, b = remove_Last_Slice_Area_Based(es,a,ed,b)
    
    if stcheck is not None:
        a,b = remove_Slice_ST_Based(a,b)
        
    if (len(a) > 2) and (len(b) > 2):
        filtered_a = dict((k, es[k]) for k in a[:-1] if k in es)
        filtered_b = dict((k, ed[k]) for k in b[:-1] if k in ed)
    
        ESV, EDV = origVolume(filtered_a,filtered_b)
    
        if STway == 'sub':
            ESVi = (es[a[-2]]['zmin'] + es[a[-1]]['zmin']) * ((abs(a[-2]-a[-1]-es[a[-1]]['minST']))/2)
            EDVi = (ed[b[-2]]['zmax'] + ed[b[-1]]['zmax']) * ((abs(b[-2]-b[-1]-ed[b[-1]]['maxST']))/2)
        if STway == 'add':
            ESVi = (es[a[-2]]['zmin'] + es[a[-1]]['zmin']) * ((abs(a[-2]-a[-1]))/2)
            ESVi = ESVi + (es[a[-1]]['zmin']*es[a[-1]]['minST'])
            EDVi = (ed[b[-2]]['zmax'] + ed[b[-1]]['zmax']) * ((abs(b[-2]-b[-1]))/2)
            EDVi = EDVi + (ed[b[-1]]['zmax']*ed[b[-1]]['maxST'])
        ESV = ESV + (ESVi / 1000)
        EDV = EDV + (EDVi / 1000)
    else:
        if len(a) == 2:
            for i in range(len(a)-1):
                ESVi = (es[a[i]]['zmin'] + es[a[i+1]]['zmin']) * ((abs(a[i]-a[i+1]-es[a[i+1]]['minST']))/2)
            ESV = (ESV + ESVi)/1000
        if len(b) == 2:
            for i in range(len(b)-1):
                EDVi = (ed[b[i]]['zmax'] + ed[b[i+1]]['zmax']) * ((abs(b[i]-b[i+1]-ed[b[i+1]]['maxST']))/2)
            EDV = (EDV + EDVi)/1000
        if len(a) == 1:
            ESV = (es[a[0]]['zmin']*es[a[0]]['minST'])/1000
        if len(b) == 1:
            EDV = (ed[b[0]]['zmax']*ed[b[0]]['maxST'])/1000
    
    return (ESV, EDV)

def zerosVolume(es,ed, issues,ty = None, rls = None, stcheck = None):
    """
    Function that looks to see if there are any zero predictions for a slice and removes that slice from the volume calculation

    Args:
      es: Dictionary of the ESFs with slice locations as keys and areas as values
      ed: Dictionary for the EDFs with slice locations as keys and areas as values
      issues: The issues dictionary that has the minframes and maxframes
      ty:  (Default value = None) Flag to send the volume calculation to STVolume
      rls:  (Default value = None) Flag to determine if to call removal of the last slice based on area
      stcheck:  (Default value = None) Flag to determine if to to remove slices based on the slice locations 


    Returns:
      ESV, EDV, and issues for each patient

    """
    ESV = 0
    EDV = 0
    a = sorted(es)
    count = 0 
    if len(a) == 2:
        for i in range(len(a)):
            keys = list(es[a[i]]['zcounts'].keys())
            lst = [int(x) for x in keys if int(x) >0]
            if len(lst) > 0:
                secmin = min(lst)
                secminframe = es[a[i]]['zcounts'][str(secmin)]['frame']
                es[a[i]].update({'zmin':secmin})
                for j in issues:
                    for k in range(len(issues[j]['minframes'])):
                        if a[i] == issues[j]['minframes'][k][2]:
                            issues[j]['minframes'][k][1] = secminframe
    else:
        for i in range(len(a)):
            count +=1   
            if (es[a[i]]['zmin'] == 0):
                    keys = list(es[a[i]]['zcounts'].keys())
                    lst = [int(x) for x in keys if int(x) >0]
                    if len(lst) > 0:
                        secmin = min(lst)
                        secminframe = es[a[i]]['zcounts'][str(secmin)]['frame']
                        es[a[i]].update({'zmin':secmin})
                        for j in issues:
                            for k in range(len(issues[j]['minframes'])):
                                if a[i] == issues[j]['minframes'][k][2]:
                                    issues[j]['minframes'][k][1] = secminframe
                    else:
                        del es[a[i]]
    b = sorted(ed)
    for i in range(len(b)):
        if ed[b[i]]['zmax'] == 0:
            del ed[b[i]]
    
    if ty is None:
        ESV, EDV = origVolume(es, ed, rls, stcheck)
    if ty == 'ST': 
        ESV, EDV = STVolume(es,ed, rls, stcheck)
    return (ESV, EDV, issues)

def getSliceOutliers(Volumes, data = None):
    """
    Function to get the patients who's records have less than five slices

    Args:
      Volumes: Volumes Dictionary outputed from computeVolume
      data:  (Default value = None) Train, Test or Validate - DSB dataset

    Returns:
      return the list of patients that have less than 5 slices

    """
    few_slices = []
    few_slices_test = []
    for i in Volumes:
        if data is None:
            if ('train') in i:
                p = i.strip('train_')
                if Volumes[i]['issues'][p]['numSlices'] < 5:
                    few_slices.append(p)
            if ('validate') in i: 
                p = i.strip('validate_')
                if Volumes[i]['issues'][p]['numSlices'] < 5:
                    few_slices.append(p)
#             if ('test') in i:
#                 p = i.strip('test_')
#                 if Volumes[i]['issues'][p]['numSlices']<5:
#                     few_slices.append(p)
        elif data == 'train':
            p = i.strip('train_')
            if Volumes[i]['issues'][p]['numSlices'] < 5:
                few_slices.append(p)
        elif data == 'validate':
            p = i.strip('validate_')
            if Volumes[i]['issues'][p]['numSlices'] < 5:
                few_slices.append(p)
        elif data == 'test':
            if ('test') in i:
                p = i.strip('test_')
                if Volumes[i]['issues'][p]['numSlices'] < 5:
                    few_slices_test.append(p)
            else:
                continue
    if data == 'test':
        return few_slices_test
    else:
        return few_slices

def getAge(origpath, agepath):
    """
    Function to get the patients age

    Args:
      origpath: original path of the dsb image
      agepath: path for the specific patient image

    Returns:
      Patient's age

    """
    img= dicomwrapper(origpath, agepath)
    age = img.PatientAge
    return age

def getGender(origpath, agepath):
    """
    Function to get the patients gender

    Args:
      origpath: original path of the dsb image
      agepath: path for the specific patient image

    Returns:
      Patient's gender

    """
    img = dicomwrapper(origpath, agepath)
    gender = img.PatientSex
    return gender

def estimateOutliersVolume(df, few_slices, data):
    """
    Function to use the gender and age model in order to estimate the volumes of the patients with less than five slices

    Args:
      df: dataframe with the volumes for each patient
      few_slices: list of patients who have less than five slices
      data: Train, Test or Validate - DSB dataset

    Returns:
      returns a dataframe of the volumes

    """
#     few_slices = getSliceOutliers(Volumes, data)
    df=df.copy()
    print (df)
    for i in range(len(few_slices)):
        print (few_slices[i])
        print (df.loc[few_slices[i]])
        age = int(df.loc[few_slices[i]]['Age_Int'])
        if df.loc[few_slices[i]]['Age Unit'] == 'M':
            age = age / 12
        gender = df.loc[few_slices[i]]['Gender']
        if (age < 16) and (gender == 'F'):
            new_sys = (2.41 * age) + 15
            new_dia = (7.61 * age) + 22
            new_ef = (new_dia - new_sys) / new_dia
            df.set_value(few_slices[i], 'Systole_P', new_sys, takeable = False)
            df.set_value(few_slices[i], 'Diastole_P', new_dia, takeable = False)
            if data is None:
                df.set_value(few_slices[i], 'EF_P', new_ef, takeable = False)
        if (age >= 16) and (gender == 'F'):
            new_sys = 53.6
            new_dia = 144
            new_ef = (new_dia - new_sys) / new_dia
            df.set_value(few_slices[i], 'Systole_P', new_sys, takeable = False)
            df.set_value(few_slices[i], 'Diastole_P', new_dia, takeable = False)
            if data is None:
                df.set_value(few_slices[i], 'EF_P', new_ef, takeable = False)
        if (age < 16) and (gender == 'M'):
            new_sys = (4.69 * age)
            new_dia = (10.8 * age) + 9
            new_ef = (new_dia - new_sys) / new_dia
            df.set_value(few_slices[i], 'Systole_P', new_sys, takeable = False)
            df.set_value(few_slices[i], 'Diastole_P', new_dia, takeable = False)
            if data is None:
                df.set_value(few_slices[i], 'EF_P', new_ef, takeable = False)
        if (age >= 16) and (gender == 'M'):
            new_sys = 75
            new_dia = 181
            new_ef = (new_dia - new_sys) / new_dia
            df.set_value(few_slices[i], 'Systole_P', new_sys, takeable = False)
            df.set_value(few_slices[i], 'Diastole_P', new_dia, takeable = False)
            if data is None:
                df.set_value(few_slices[i], 'EF_P', new_ef, takeable = False)
    print (df.loc[few_slices[0]])
    print (df.loc[few_slices[1]])
    return df
def removeOutliers_Update(df, Volumes, data):
    """
    Function that calls estimateOutliersVolume

    Args:
      df: dataframe with the volumes for each patient
      few_slices: list of patients who have less than five slices
      data: Train, Test or Validate - DSB dataset

    Returns:
      returns a dataframe of the volumes

    """
    few_slices = getSliceOutliers(Volumes, data)
    df = df.copy()
    tmp = []
    idx = []
    print (few_slices)
    for i in range(len(few_slices)):
#         print (df.loc[int(few_slices[i])])
        t = df.loc[int(few_slices[i])]
        print (type(t), t.index, t.values)
        idx = t.index
        tmp.append(t.values)
#         tmp = removeLowEstimates(t)
#         print (tmp.index)
        df = df.drop(int(few_slices[i]))
    print (tmp, idx, few_slices)
    tmp_df = pd.DataFrame(tmp, index = few_slices, columns = idx)
#     print (tmp_df)
    new_df = estimateOutliersVolume(tmp_df, few_slices, data)
    print (new_df)
    right_df = pd.concat([df, new_df], axis=0)
    print (right_df.loc[few_slices[0]])
    return right_df
                          
def removeOutliers(df, Volumes, data):
    """
    Function that drops the patients who have less than five slices from the dataframe

    Args:
      df: dataframe with the volumes for each patient
      few_slices: list of patients who have less than five slices
      data: Train, Test or Validate - DSB dataset

    Returns:
      returns a dataframe of the volumes

    """
    few_slices = getSliceOutliers(Volumes, data)
    df = df
    for i in range(len(few_slices)):
        df = df.drop(int(few_slices[i]))
    return df
    
def removeNoVolume(df):
    """
    Remove the patients with no volumes

    Args:
      df: dataframe with the volumes for each patient

    Returns:
      dataframe with patient volumes

    """
    tmp = df
    all_values = tmp[(tmp['Systole_P'] > 0) & (tmp['Diastole_P'] >0)]

    return all_values

def getbest(df, actual, predicted, dev):
    """
    Function to get the list of patients that have RMSEs within the provided range

    Args:
      df: dataframe of the patient volumes
      actual: Systole, Diastole, or EF
      predicted: Systole_P, Diastole_P, EF_P
      dev: range for the RMSE to be within

    Returns:
      list of the top patients

    """
    diff = df[actual]-df[predicted]
#     top_low = diff.index[diff < dev].tolist()
#     top_high = diff.index[diff > -dev].tolist()
    top = diff.index[(diff > -dev) & (diff < dev)].tolist()
#     print ('**********************')
#     print (diff)
    return (top)

def removeLowHigh(df, dev, val):
    """
    Function that 

    Args:
      df: dataframe of the patient volumes
      dev: value from RMSE to be off by 
      val: Systole or Diastole

    Returns:
      Returns dataframe with the low and high off patients

    """
    col = val+'_diff'
    tmp = df
#     print (tmp.shape)
    orig = tmp.shape[0]
    low_idx = tmp[tmp[col] > dev].index.tolist()
    tmp = tmp[(tmp[col] < dev)]
#     print (tmp.shape)
    high_idx = tmp[tmp[col] < -dev].index.tolist()
    tmp = tmp[(tmp[col] > -dev)]
#     print (tmp.shape)
    new = tmp.shape[0]
    #print (tmp)
    return (tmp, orig-new, low_idx, high_idx)

def create_df(Volumes):
    """
    Creates the dataframe from the results of computeVolume

    Args:
      Volumes: dictionary of patients Volumes

    Returns:
      Dataframe of patients volumes

    """
    train_gt = pd.read_csv('/masvol/data/dsb/train.csv')
    validate_gt = pd.read_csv('/masvol/data/dsb/validate.csv')
    
    train_pred = []
    validate_pred = []

    for i in Volumes:
        if ('train') in i:
            ID = i.strip('train_')
            ESV = Volumes[i]['ESV']
            EDV = Volumes[i]['EDV']
            age = Volumes[i]['issues'][ID]['age']
            age_unit = age[-1]
            age_int = int(age[:-1])
            gender = Volumes[i]['issues'][ID]['gender']
            slices = Volumes[i]['issues'][ID]['numSlices']
#             print (slices,Volumes[i]['issues'][ID]['numSlices'] )
            train_pred.append([int(ID), ESV, EDV,age, age_int, age_unit, gender,slices])
#             print (i,ID)
#             train_pred.append([int(ID), ESV, EDV])
        if ('validate') in i:
            ID = i.strip('validate_')
            ESV = Volumes[i]['ESV']
            EDV = Volumes[i]['EDV']
            age = Volumes[i]['issues'][ID]['age']
            age_unit = age[-1]
            age_int = int(age[:-1])
            gender = Volumes[i]['issues'][ID]['gender']
            slices = Volumes[i]['issues'][ID]['numSlices']
            validate_pred.append([int(ID), ESV, EDV,age, age_int, age_unit, gender,slices])
#             print (i,ID)
#             validate_pred.append([int(ID), ESV, EDV])
    train_pred_df = pd.DataFrame(train_pred, columns= ['Id','Systole_P', 'Diastole_P',
                                                       'Age','Age_Int','Age Unit', 'Gender','NumSlices'])
    validate_pred_df = pd.DataFrame(validate_pred, columns= ['Id','Systole_P', 'Diastole_P',
                                                            'Age','Age_Int','Age Unit','Gender',
                                                             'NumSlices'])
    train_df = pd.concat([train_gt.set_index('Id'),train_pred_df.set_index('Id')], axis=1, join='inner')
    train_df['Systole_diff'] = train_df['Systole'] - train_df['Systole_P']
    train_df['Diastole_diff'] = train_df['Diastole'] - train_df['Diastole_P']
    train_df['EF_P'] = (train_df['Diastole_P']-train_df['Systole_P'])/train_df['Diastole_P']
    train_df['EF'] = (train_df['Diastole']-train_df['Systole'])/train_df['Diastole']
    validate_df = pd.concat([validate_gt.set_index('Id'),validate_pred_df.set_index('Id')], axis=1, join='inner')
    validate_df['Systole_diff'] = validate_df['Systole'] - validate_df['Systole_P']
    validate_df['Diastole_diff'] = validate_df['Diastole'] - validate_df['Diastole_P']
    validate_df['EF_P'] = (validate_df['Diastole_P']-validate_df['Systole_P'])/validate_df['Diastole_P']
    validate_df['EF'] = (validate_df['Diastole']-validate_df['Systole'])/validate_df['Diastole']
    all_df = pd.concat([train_df, validate_df], axis=0)
    
    return (train_df, validate_df, all_df)

def removeLowEstimates(all_df, data):
    """
    Uses the gender and age model to estimate the volumes for patients where the predictions are low
    These formulas are from the 2nd DSB leader gender and age model. 
    Args:
      all_df: dataframe of patient volumes
      data: test, train, validate

    Returns:

    """
    df = all_df.copy()
    sys_males_y_idx = df.loc[(df['Systole_P'] < 2.3) &
                                          (df['Gender'] == 'M') & 
                                          (df['Age_Int'] < 16)].index

    sys_fem_y_idx = df.loc[(df['Systole_P'] < 2.3) &
                                        (df['Gender']== 'F') &
                                        (df['Age_Int'] < 16)].index

    sys_males_o_idx = df.loc[(df['Systole_P'] < 2.3) &
                                          (df['Gender'] == 'M') &
                                          (df['Age_Int'] >= 16)].index
    sys_fem_o_idx = df.loc[(df['Systole_P'] < 2.3) &
                                        (df['Gender']== 'F') &
                                        (df['Age_Int'] >= 16)].index
    for i in sys_fem_y_idx:
        age = df.loc[i]['Age_Int']
        if df.loc[i]['Age Unit'] =='M':
            age = age/12
        new_vol = (2.41 * age) + 15
        ef_p = ((df.loc[i]['Diastole_P'] - new_vol) / df.loc[i]['Diastole_P'])
        df.set_value(i, 'Systole_P', new_vol, takeable=False)
        if data is None:
            df.set_value(i, 'EF_P', ef_p, takeable = False)
    for i in sys_males_y_idx:
        age = df.loc[i]['Age_Int']
        if df.loc[i]['Age Unit'] =='M':
            age = age/12
        new_vol = (4.69 * age)
        ef_p = ((df.loc[i]['Diastole_P'] - new_vol) / df.loc[i]['Diastole_P'])
        df.set_value(i, 'Systole_P', new_vol, takeable=False)
        if data is None:
            df.set_value(i, 'EF_P', ef_p, takeable = False)
    for i in sys_fem_o_idx:
        age = df.loc[i]['Age_Int']
        if df.loc[i]['Age Unit'] =='M':
            age = age/12
        new_vol = 53.6
        ef_p = ((df.loc[i]['Diastole_P'] - new_vol) / df.loc[i]['Diastole_P'])
        df.set_value(i, 'Systole_P', new_vol, takeable=False)
        if data is None:
            df.set_value(i, 'EF_P', ef_p, takeable = False)
    for i in sys_males_o_idx:
        age = df.loc[i]['Age_Int']
        if df.loc[i]['Age Unit'] =='M':
            age = age/12
        new_vol = 75
        ef_p = ((df.loc[i]['Diastole_P'] - new_vol) / df.loc[i]['Diastole_P'])
        df.set_value(i, 'Systole_P', new_vol, takeable=False)
        if data is None:
            df.set_value(i, 'EF_P', ef_p, takeable = False)

    dia_males_y_idx = df.loc[(df['Diastole_P'] < 5) &
                                          (df['Gender'] == 'M') &
                                          (df['Age_Int'] < 16)].index
    dia_fem_y_idx = df.loc[(df['Diastole_P'] < 5) &
                                        (df['Gender']== 'F') & 
                                        (df['Age_Int'] < 16)].index
    dia_males_o_idx = df.loc[(df['Diastole_P'] < 5) &
                                          (df['Gender'] == 'M') &
                                          (df['Age_Int'] >= 16)].index
    dia_fem_o_idx = df.loc[(df['Diastole_P'] < 5) &
                                        (df['Gender'] == 'F') & 
                                        (df['Age_Int'] >= 16)].index
    for i in dia_fem_y_idx:
        age = df.loc[i]['Age_Int']
        if df.loc[i]['Age Unit'] =='M':
            age = age/12
        new_vol = (7.61 * age) + 22
        ef_p = ((new_vol - df.loc[i]['Systole_P']) / new_vol)
        df.set_value(i, 'Diastole_P', new_vol, takeable=False)
        if data is None:
            df.set_value(i, 'EF_P', ef_p, takeable = False)

    for i in dia_males_y_idx:
        age = df.loc[i]['Age_Int']
        if df.loc[i]['Age Unit'] =='M':
            age = age/12
        new_vol = (10.8 * age) + 9
        ef_p = ((new_vol - df.loc[i]['Systole_P']) / new_vol)
        df.set_value(i, 'Diastole_P', new_vol, takeable=False)
        if data is None:
            df.set_value(i, 'EF_P', ef_p, takeable = False)

    for i in dia_fem_o_idx:
        age = df.loc[i]['Age_Int']
        if df.loc[i]['Age Unit'] =='M':
            age = age/12
        new_vol = 144
        ef_p = ((new_vol - df.loc[i]['Systole_P']) / new_vol)
        df.set_value(i, 'Diastole_P', new_vol, takeable=False)
        if data is None:
            df.set_value(i, 'EF_P', ef_p, takeable = False)

    for i in dia_males_o_idx:
        age = df.loc[i]['Age_Int']
        if df.loc[i]['Age Unit'] =='M':
            age = age/12
        new_vol = 181
        ef_p = ((new_vol - df.loc[i]['Systole_P']) / new_vol)
        df.set_value(i, 'Diastole_P', new_vol, takeable=False)
        if data is None:
            df.set_value(i, 'EF_P', ef_p, takeable = False)

    print (df.columns)
    return df

def removeLowEstimatesDSBLR(all_df, data):
    """
    Uses the gender and age model to estimate the volumes for patients where the predictions are low.
    These formulas are from the DSB Train dataset that we derived. 

    Args:
      all_df: dataframe of patient volumes
      data: test, train, validate

    Returns:

    """
    df = all_df.copy()
    sys_males_y_idx = df.loc[(df['Systole_P'] < 2.3) &
                                          (df['Gender'] == 'M') & 
                                          (df['Age_Int'] < 16)].index

    sys_fem_y_idx = df.loc[(df['Systole_P'] < 2.3) &
                                        (df['Gender']== 'F') &
                                        (df['Age_Int'] < 16)].index

    sys_males_o_idx = df.loc[(df['Systole_P'] < 2.3) &
                                          (df['Gender'] == 'M') &
                                          (df['Age_Int'] >= 16)].index
    sys_fem_o_idx = df.loc[(df['Systole_P'] < 2.3) &
                                        (df['Gender']== 'F') &
                                        (df['Age_Int'] >= 16)].index
    for i in sys_fem_y_idx:
        age = df.loc[i]['Age_Int']
        if df.loc[i]['Age Unit'] =='M':
            age = age/12
        new_vol = (1.7 * age) + 26
        ef_p = ((df.loc[i]['Diastole_P'] - new_vol) / df.loc[i]['Diastole_P'])
        df.set_value(i, 'Systole_P', new_vol, takeable=False)
        if data is None:
            df.set_value(i, 'EF_P', ef_p, takeable = False)
    for i in sys_males_y_idx:
        age = df.loc[i]['Age_Int']
        if df.loc[i]['Age Unit'] =='M':
            age = age/12
        new_vol = (3.7 * age) + 6.8
        ef_p = ((df.loc[i]['Diastole_P'] - new_vol) / df.loc[i]['Diastole_P'])
        df.set_value(i, 'Systole_P', new_vol, takeable=False)
        if data is None:
            df.set_value(i, 'EF_P', ef_p, takeable = False)
    for i in sys_fem_o_idx:
        age = df.loc[i]['Age_Int']
        if df.loc[i]['Age Unit'] =='M':
            age = age/12
        new_vol = (0.11 * age) + 79.1
        ef_p = ((df.loc[i]['Diastole_P'] - new_vol) / df.loc[i]['Diastole_P'])
        df.set_value(i, 'Systole_P', new_vol, takeable=False)
        if data is None:
            df.set_value(i, 'EF_P', ef_p, takeable = False)
    for i in sys_males_o_idx:
        age = df.loc[i]['Age_Int']
        if df.loc[i]['Age Unit'] =='M':
            age = age/12
        new_vol = (0.09 * age) + 58
        ef_p = ((df.loc[i]['Diastole_P'] - new_vol) / df.loc[i]['Diastole_P'])
        df.set_value(i, 'Systole_P', new_vol, takeable=False)
        if data is None:
            df.set_value(i, 'EF_P', ef_p, takeable = False)

    dia_males_y_idx = df.loc[(df['Diastole_P'] < 5) &
                                          (df['Gender'] == 'M') &
                                          (df['Age_Int'] < 16)].index
    dia_fem_y_idx = df.loc[(df['Diastole_P'] < 5) &
                                        (df['Gender']== 'F') & 
                                        (df['Age_Int'] < 16)].index
    dia_males_o_idx = df.loc[(df['Diastole_P'] < 5) &
                                          (df['Gender'] == 'M') &
                                          (df['Age_Int'] >= 16)].index
    dia_fem_o_idx = df.loc[(df['Diastole_P'] < 5) &
                                        (df['Gender'] == 'F') & 
                                        (df['Age_Int'] >= 16)].index
    for i in dia_fem_y_idx:
        age = df.loc[i]['Age_Int']
        if df.loc[i]['Age Unit'] =='M':
            age = age/12
        new_vol = (7 * age) + 36
        ef_p = ((new_vol - df.loc[i]['Systole_P']) / new_vol)
        df.set_value(i, 'Diastole_P', new_vol, takeable=False)
        if data is None:
            df.set_value(i, 'EF_P', ef_p, takeable = False)

    for i in dia_males_y_idx:
        age = df.loc[i]['Age_Int']
        if df.loc[i]['Age Unit'] =='M':
            age = age/12
        new_vol = (10 * age) + 12.5
        ef_p = ((new_vol - df.loc[i]['Systole_P']) / new_vol)
        df.set_value(i, 'Diastole_P', new_vol, takeable=False)
        if data is None:
            df.set_value(i, 'EF_P', ef_p, takeable = False)

    for i in dia_fem_o_idx:
        age = df.loc[i]['Age_Int']
        if df.loc[i]['Age Unit'] =='M':
            age = age/12
        new_vol = (-0.3 * age) + 204
        ef_p = ((new_vol - df.loc[i]['Systole_P']) / new_vol)
        df.set_value(i, 'Diastole_P', new_vol, takeable=False)
        if data is None:
            df.set_value(i, 'EF_P', ef_p, takeable = False)

    for i in dia_males_o_idx:
        age = df.loc[i]['Age_Int']
        if df.loc[i]['Age Unit'] =='M':
            age = age/12
        new_vol = (-0.1 * age) + 156.4
        ef_p = ((new_vol - df.loc[i]['Systole_P']) / new_vol)
        df.set_value(i, 'Diastole_P', new_vol, takeable=False)
        if data is None:
            df.set_value(i, 'EF_P', ef_p, takeable = False)

    print (df.columns)
    return df

def estimateCheck(df, actual, predicted, dev):
    """
    Function to determine which patients RMSE are high or low

    Args:
      df: patient volume dataframe
      actual: Systole, Diastole, EF
      predicted: Systole_P, Diastole_P, EF_P
      dev: deviation from the correct RMSE

    Returns:

    """
    diff = df[actual]-df[predicted]
#     print (diff)
    if dev == 0: 
        low = diff[diff > 0].count()
        correct = diff[diff == 0].count()
        high = diff[diff < 0].count()
        print (actual,': % Low Estimate:', round((low/len(df)),2)*100, '% Correct Estimate: ',
               round((correct/len(df)),2)*100,'% High Estimate: ', (round((high/len(df)),2))*100)
        
    else:
        low = diff[diff > dev].count()
        correct = diff[(diff < 5) & (-5 < diff)].count()
        high = diff[diff < -dev].count()
        print (actual,': % Low Estimate:', round((low/len(df)),2)*100, '% Correct Estimate: ',
               round((correct/len(df)),2)*100,'% High Estimate: ', (round((high/len(df)),2))*100)
        print ('Lows: ', diff.index[diff > dev].tolist())
        print ('Highs: ', diff.index[diff < -dev].tolist())
        low = round((low/len(df)),2)*100
        high = round((high/len(df)),2)*100
        return (low, high,diff.index[diff > dev].tolist(),diff.index[diff < -dev].tolist())
#         return (low, high,diff.index[diff < dev].tolist(),diff.index[diff > -dev].tolist())

def plot_outlier_imgs(img_file, pred_file, method, Type,img_txt, v, image, frames):
    """
    Function to Plot the images

    Args:
      img_file: image files
      pred_file: prediction files
      method: normalization method
      Type: normalization type
      img_txt: text file that lists the frames locations in the 4-D array
      v: volume dataframe
      image: list of images
      frames: list of frames

    Returns:
      Calls display_images_predictions

    """
    idx = []
    
    f = open(img_txt, 'r')
    x = f.readlines()
    f.close()
    imgs = x[0].split(',')
    orig = []

    if int(image) <=500:
        t = 'train_'+image
        s = 'train'
    if (int(image) > 500) & (int(image) <= 700):
        t = 'validate_'+image
        s = 'validate'
    if (int(image) > 700) :
        t = 'test_'+image
        s = 'test'
    f = v[t]['issues'][image][frames]

    base_dir = '/masvol/output/dsb/norm/'
    f = sorted(f, key=lambda frame: frame[2])
    for i in range(len(f)):

        if isinstance(f[i][1], list):

            f_0 = f[i][0]
            f_1 = f[i][1][0]
        else:

            f_0 = f[i][0]
            f_1 = f[i][1]
        img_f_t = base_dir+method+'/'+Type+'/'+s+'/'+image+'/'+f_0+'_'+f_1

        img_o_t = base_dir+method+'/'+Type+'/'+s+'/'+image+'/'+f_0+'_'+f_1

        idx.append(imgs.index(img_f_t))
        orig.append(img_o_t)

    display_images_predictions (img_file, pred_file, image_list = idx, orig=orig)

def plotting (df, actual, predicted):
    """
      Function to plot the actual vs predicted 

    Args:
      df: volume dataframe
      actual: Systole, Diastole, EF
      predicted: Systole_P, Diastole_P, EF_P

    Returns:
      plot fo the actual vs predicted 

    """
    plt.figure(figsize=(8, 6))
    plt.scatter(df[predicted], df[actual],)
    plt.plot(range(len(df[predicted])), range(len(df[actual])), color = 'green')
    plt.xlabel(predicted)
    plt.ylabel(actual)
    plt.xlim(0,600)
    plt.ylim(0, 600)
    
def display_images_predictions (image_file, pred_file,  num_images=4, image_list=False, random_images=False, orig=None):
    """
    Function to display images,predictions and overlays of images and predictions.

    Args:
      image_file(: string):  image file (.npy) with full path.
      pred_file(: string):  prediction file (.npy) with full path.
      image_list (: list, optional) : list images to be displayed, if this field is present then num_images and random_images will be ignored.
      num_images (: int, optional) : number of images to be displayed, default is 4.
      random_images (: boolean, optional) : if True pick images randomly, else display first n images, default is False.
      image_file: 
      pred_file: 
      num_images:  (Default value = 4)
      image_list:  (Default value = False)
      random_images:  (Default value = False)
      orig:  (Default value = None)

    Returns:
      : None.
      Note:
      prection file should have the sigmoid outputs (not the rounded values).

    """
    ts = np.load(image_file)
    pred = np.load(pred_file)
    samples, x, y, z = pred.shape
    print ("samples, max, min ", samples, pred.max(), pred.min())
    pred2 = np.round(pred)

    display_list = []
    if image_list == False:
        if random_images == True:
            display_list = random.sample(range(0, samples), num_images)
        else :
            display_list = [i for i in range (num_images)]
    else:
        display_list = image_list
    count = 0
    for i in display_list:
        if orig is not None:
            print (orig[count])
        f, axs = plt.subplots(1,3,figsize=(15,15))
        print (i)
        plt.subplot(131),plt.imshow(ts[i].reshape(x, y))
        plt.title('Image '+str(i)), plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(pred2[i].reshape(x, y))
        plt.title('Prediction'), plt.xticks([]), plt.yticks([])
        plt.subplot(133),plt.imshow(ts[i].reshape(x, y)), plt.imshow(pred2[i].reshape(x, y), 'binary', interpolation='none', alpha=0.3)
        plt.title('Overlay'), plt.xticks([]), plt.yticks([])
        plt.show()
        count +=1

def compute_rmse(df, source, val = None):
    """
      Function to compute the RMSE for the volumes

    Args:
      df: dataframe of patient volume
      source: test, train, validate
      val:  (Default value = None) D - Diastole, S - Systole, E - EF

    Returns:
      Returns the Diastole RMSE, Systole RMSE, and EF RMSE

    """
    if val is None:
        sys = np.sqrt(sklearn.metrics.mean_squared_error(df['Systole'],df['Systole_P']))
        dia = np.sqrt(sklearn.metrics.mean_squared_error(df['Diastole'],df['Diastole_P']))
        ef = np.sqrt(sklearn.metrics.mean_squared_error(df['EF'],df['EF_P']))
#         print (source,': Systole RMSE: ', sys, 'Diastole RMSE: ', dia, ': EF RMSE: ', ef)
        print (source,": ","Systole RMSE: %.2f" % round(sys,2), "ml ", "Diastole RMSE: '%.2f" % round(dia,2),
               "ml ","EF RMSE: %.2f" % (round(ef,2)*100), "%")
        return (round(sys,2),round(dia,2), (round(ef,2)*100))
    elif ('D') in val:
        dia = np.sqrt(sklearn.metrics.mean_squared_error(df['Diastole'],df['Diastole_P']))
        print (source,'Diastole RMSE: ', dia)
    elif ('S') in val:
        sys = np.sqrt(sklearn.metrics.mean_squared_error(df['Systole'],df['Systole_P']))
        print (source,': Systole RMSE: ', sys)
    elif ('E') in val:
        ef = np.sqrt(sklearn.metrics.mean_squared_error(df['EF'],df['EF_P']))
        print (source, ': EF RMSE: ', ef)
        
        
