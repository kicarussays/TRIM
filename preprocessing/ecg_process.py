import pandas as pd
import numpy as np
import os

alldf = pd.read_csv('../../usedata/snuh/ecg_index_time.csv')

import ray
from tqdm import tqdm
# ray.init(num_cpus=64)

import xml
import xml.etree.ElementTree as ET
def xml_to_dict(element):
    result = {}
    
    # 속성이 있으면 추가
    if element.attrib:
        result.update(element.attrib)
    
    # 자식 요소들 처리
    for child in element:
        child_data = xml_to_dict(child)
        
        # 같은 태그명이 여러 개 있는 경우 리스트로 처리
        if child.tag in result:
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_data)
        else:
            result[child.tag] = child_data
    
    # 텍스트 내용이 있으면 추가
    if element.text and element.text.strip():
        if result:
            result['text'] = element.text.strip()
        else:
            return element.text.strip()
    
    return result

# @ray.remote
def process_file(flist):
    all_stmt = []
    for filepath in tqdm(flist):
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            rdict = xml_to_dict(root)
            
            if type(rdict['Diagnosis']['DiagnosisStatement']) == list:
                all_stmt.append([filepath, np.array([i['StmtText'] for i in rdict['Diagnosis']['DiagnosisStatement']])])
            else:
               all_stmt.append([filepath, np.array([rdict['Diagnosis']['DiagnosisStatement']['StmtText']])])
        except:
            print('error')
            all_stmt.append([filepath, np.array(['None'])])

    return all_stmt


import numpy as np
import base64
import struct

def convert_ecg_to_mimic_format(ecg_data):
    """
    ECG XML 데이터를 MIMIC-IV 순서에 맞게 12x5000 numpy array로 변환
    
    MIMIC-IV ECG 리드 순서: I, II, III, aVF, aVR, aVL, V1, V2, V3, V4, V5, V6
    """
    
    # Waveform 데이터 추출
    if 'Waveform' in ecg_data:
        waveform_data = ecg_data['Waveform']
    else:
        raise ValueError("Cannot find Waveform data in the input")
    
    # print(f"Found {len(waveform_data)} waveform(s)")
    
    # MIMIC-IV 순서 정의
    mimic_lead_order = ['I', 'II', 'III', 'aVF', 'aVR', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # 결과 배열 초기화 (12 leads x 5000 samples)
    result_array = np.zeros((12, 5000), dtype=np.float32)
    
    # Rhythm 데이터 처리 (더 긴 10초 데이터 사용)
    rhythm_waveform = None
    median_waveform = None
    
    for waveform in waveform_data:
        if waveform['WaveformType'] == 'Rhythm':
            rhythm_waveform = waveform
        elif waveform['WaveformType'] == 'Median':
            median_waveform = waveform
    
    # Rhythm 데이터가 있으면 우선 사용, 없으면 Median 데이터 사용
    target_waveform = rhythm_waveform if rhythm_waveform else median_waveform
    
    if not target_waveform:
        raise ValueError("No valid waveform data found")
    
    # 현재 데이터에서 사용 가능한 리드들 매핑
    available_leads = {}
    for lead_data in target_waveform['LeadData']:
        lead_id = lead_data['LeadID']
        
        # Base64 인코딩된 파형 데이터 디코딩
        waveform_data = base64.b64decode(lead_data['WaveFormData'])
        
        # 2바이트 정수로 언패킹 (little-endian)
        samples = struct.unpack(f'<{len(waveform_data)//2}h', waveform_data)
        
        # 마이크로볼트 단위로 변환
        amplitude_per_bit = float(lead_data['LeadAmplitudeUnitsPerBit'])
        voltage_data = np.array(samples) * amplitude_per_bit / 1000.0  # mV로 변환
        
        available_leads[lead_id] = voltage_data
    
    # print(f"Available leads: {list(available_leads.keys())}")
    # print(f"Sample counts: {[len(data) for data in available_leads.values()]}")
    
    # MIMIC-IV 순서에 맞게 데이터 배치
    for i, target_lead in enumerate(mimic_lead_order):
        if target_lead in available_leads:
            # 데이터가 있는 경우 직접 사용
            lead_data = available_leads[target_lead]
            
            # 5000 샘플에 맞게 조정
            if len(lead_data) >= 5000:
                result_array[i, :] = lead_data[:5000]
            else:
                # 데이터가 부족한 경우 제로 패딩
                result_array[i, :len(lead_data)] = lead_data
                
        elif target_lead in ['III', 'aVF', 'aVR', 'aVL']:
            # 계산 가능한 리드들 계산
            if target_lead == 'III' and 'I' in available_leads and 'II' in available_leads:
                # Lead III = Lead II - Lead I
                lead_i = available_leads['I'][:5000] if len(available_leads['I']) >= 5000 else available_leads['I']
                lead_ii = available_leads['II'][:5000] if len(available_leads['II']) >= 5000 else available_leads['II']
                min_len = min(len(lead_i), len(lead_ii), 5000)
                result_array[i, :min_len] = lead_ii[:min_len] - lead_i[:min_len]
                
            elif target_lead == 'aVF' and 'II' in available_leads and 'III' in available_leads:
                # aVF = (Lead II + Lead III) / 2
                lead_ii = available_leads['II'][:5000] if len(available_leads['II']) >= 5000 else available_leads['II']
                # Lead III 계산 또는 직접 사용
                if 'III' in available_leads:
                    lead_iii = available_leads['III'][:5000] if len(available_leads['III']) >= 5000 else available_leads['III']
                else:
                    lead_i = available_leads['I'][:5000] if len(available_leads['I']) >= 5000 else available_leads['I']
                    lead_iii = lead_ii[:min(len(lead_ii), len(lead_i))] - lead_i[:min(len(lead_ii), len(lead_i))]
                
                min_len = min(len(lead_ii), len(lead_iii), 5000)
                result_array[i, :min_len] = (lead_ii[:min_len] + lead_iii[:min_len]) / 2
                
            elif target_lead == 'aVR' and 'I' in available_leads and 'II' in available_leads:
                # aVR = -(Lead I + Lead II) / 2
                lead_i = available_leads['I'][:5000] if len(available_leads['I']) >= 5000 else available_leads['I']
                lead_ii = available_leads['II'][:5000] if len(available_leads['II']) >= 5000 else available_leads['II']
                min_len = min(len(lead_i), len(lead_ii), 5000)
                result_array[i, :min_len] = -(lead_i[:min_len] + lead_ii[:min_len]) / 2
                
            elif target_lead == 'aVL' and 'I' in available_leads and 'III' in available_leads:
                # aVL = (Lead I - Lead III) / 2
                lead_i = available_leads['I'][:5000] if len(available_leads['I']) >= 5000 else available_leads['I']
                # Lead III 계산 또는 직접 사용
                if 'III' in available_leads:
                    lead_iii = available_leads['III'][:5000] if len(available_leads['III']) >= 5000 else available_leads['III']
                else:
                    lead_ii = available_leads['II'][:5000] if len(available_leads['II']) >= 5000 else available_leads['II']
                    lead_iii = lead_ii[:min(len(lead_ii), len(lead_i))] - lead_i[:min(len(lead_ii), len(lead_i))]
                
                min_len = min(len(lead_i), len(lead_iii), 5000)
                result_array[i, :min_len] = (lead_i[:min_len] - lead_iii[:min_len]) / 2
        
        else:
            print(f"Warning: {target_lead} lead data not available and cannot be calculated")
    
    return result_array


import os
import modin.pandas as mpd
import ray
import wfdb
# ray.init(num_cpus=64)


from scipy.signal import butter, lfilter
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter_and_normalization(data, lowcut=0.05, highcut=150, fs=500, order=5, normalization=True):
    yall = []
    for dat in data:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        bp = lfilter(b, a, dat)
        if normalization:
            # Normalize between -1 and 1.ipynb
            y = 2*(bp - np.min(bp)) / ((np.max(bp)-np.min(bp)) + 1e-5)
            bp = y - 1
        yall.append(bp)
    
    return np.array(yall)

# 멀티프로세싱으로 변경
def process_ecg_signal(path):
    """단일 ECG 신호를 처리하는 함수"""
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        rdict = xml_to_dict(root)
        ecg_array = convert_ecg_to_mimic_format(rdict)
        ecg_signals = butter_bandpass_filter_and_normalization(
           np.nan_to_num(ecg_array, nan=0))  # 12 * 5000
        return ecg_signals
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
if __name__ == '__main__':
    # CPU 코어 수에 따라 프로세스 수 결정 (또는 원하는 수로 설정)
    num_processes = 80  # 또는 원하는 프로세스 수로 설정 (예: 32)
    
    print(f"Using {num_processes} processes")
    
    with Pool(processes=num_processes) as pool:
        # tqdm을 사용해서 진행상황 표시
        all_signals = list(tqdm(
            pool.imap(process_ecg_signal, alldf['study_id']), 
            total=len(alldf),
            desc="Processing ECG signals"
        ))
    
    # None 값 제거 (에러가 발생한 경우)
    all_signals = [signal for signal in all_signals if signal is not None]
    
    print(f"Successfully processed {len(all_signals)} signals")
all_signals = np.stack(all_signals)

np.save('../../usedata/snuh/all_signals.npy', all_signals)