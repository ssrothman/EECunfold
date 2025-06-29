import datasets
import re
import os

def parse_loss_name(name):
    splitted = name.split('_')
    tag = '_'.join(splitted[:3])
    samplesplits = []
    for i in range(3, len(splitted)):
        if splitted[i].startswith('boot'):
            break
        samplesplits.append(splitted[i])
    sample = '_'.join(samplesplits)
    import re
    m = re.search(r'_boot(\d+)', name)
    if m:
        nboot = int(m.group(1))
    else:
        nboot = 0
    m = re.search(r'_(\d+)stat(\d+)', name)
    if m:
        statN = int(m.group(1))
        statK = int(m.group(2))
    else:
        statN = -1
        statK = -1
    m = re.search(r'_first(\d+)', name)
    if m:
        firstN = int(m.group(1))
    else:
        firstN = -1
    syst_l = []
    m = re.search(r'_SYST(?:-([a-zA-Z0-9_]+))*', name)
    if m:
        syst_l = m.group(0).split('-')[1:]
    testcut = '_TESTCUT' in name
    return tag, sample, nboot, statN, statK, firstN, syst_l, testcut

def loss_name(tag, sample, nboot, statN, statK, firstN, syst_l, testcut):
    if nboot < 0:
        options = os.listdir(os.path.join(datasets.basedir, tag, sample, 'EECres4tee', 'CONSTRUCTED_LOSSES'))
        options = list(filter(lambda x: x.startswith(f'{tag}_{sample}_'), options))
        if statN > 0:
            options = list(filter(lambda x: f'_{statN}stat{statK}' in x, options))
        else:
            options = list(filter(lambda x: 'stat' not in x, options))
        if firstN > 0:
            options = list(filter(lambda x: f'_first{firstN}' in x, options))
        if len(syst_l) > 0:
            syststr = '_SYST'
            for syst in syst_l:
                syststr += f'-{syst}'
            options = list(filter(lambda x: syststr in x, options))
        if testcut:
            options = list(filter(lambda x: '_TESTCUT' in x, options))
        if len(options) == 0:
            raise ValueError(f"No options found for tag {tag}, sample {sample}, firstN {firstN}, syst_l {syst_l}, testcut {testcut}.")
        elif len(options) > 1:
            print("Warning: multiple options for nboot found for loss:")
            nboot_options = []
            for i, opt in enumerate(options):
                m = re.search(r'_boot(\d+)', opt)
                if m:
                    nboot_options.append(int(m.group(1)))
                else:
                    nboot_options.append(0)
                print("\t%d: nboot=%d"%(i, nboot_options[-1]))
            print("Using the largest option. If this is not what you want, please specify nboot explicitly.")
            nboot = max(nboot_options)
        else:
            m = re.search(r'_boot(\d+)', options[0])
            if m:
                nboot = int(m.group(1))
            else:
                nboot = 0

    name = f'{tag}_{sample}'
    name += f'_boot{nboot}'
    if statN > 0:
        name += f'_{statN}stat{statK}'
    if firstN > 0:
        name += f'_first{firstN}'
    if len(syst_l) > 0:
        name += '_SYST'
        for syst in syst_l:
            name += f'-{syst}'
    if testcut:
        name += '_TESTCUT'
    return name

def loss_folder(tag, sample, nboot, statN, statK, firstN, syst_l, testcut):
    name = loss_name(tag, sample, nboot, statN, statK, firstN, syst_l, testcut)

    path = os.path.join(
        datasets.basedir, tag, sample,
        'EECres4tee', 'CONSTRUCTED_LOSSES',
        name)
    return path

def parse_reco_name(name):
    splitted = name.split('_')
    tag = '_'.join(splitted[:3])
    samplesplits = []
    for i in range(3, len(splitted)):
        if splitted[i].startswith('boot'):
            break
        samplesplits.append(splitted[i])
    sample = '_'.join(samplesplits)
    import re
    m = re.search(r'_boot(\d+)', name)
    if m:
        nboot = int(m.group(1))
    else:
        nboot = 0
    m = re.search(r'_(\d+)stat(\d+)', name)
    if m:
        statN = int(m.group(1))
        statK = int(m.group(2))
    else:
        statN = -1
        statK = -1
    m = re.search(r'_first(\d+)', name)
    if m:
        firstN = int(m.group(1))
    else:
        firstN = -1
    objsyst = splitted[-2]
    wtsyst = splitted[-1]
    testcut = '_TESTCUT' in name
    return tag, sample, nboot, statN, statK, firstN, objsyst, wtsyst, testcut

def reco_name(tag, sample, nboot, statN, statK, firstN, objsyst, wtsyst, testcut):
    if nboot < 0:
        options = os.listdir(os.path.join(datasets.basedir, tag, sample, 'EECres4tee', 'CONSTRUCTED_RECO')) 
        options = list(filter(lambda x: x.startswith(f'{tag}_{sample}_'), options))
        if statN > 0:
            options = list(filter(lambda x: f'_{statN}stat{statK}' in x, options))
        else:
            options = list(filter(lambda x: 'stat' not in x, options))
        if firstN > 0:
            options = list(filter(lambda x: f'_first{firstN}' in x, options))
        if testcut:
            options = list(filter(lambda x: '_TESTCUT' in x, options))
        if len(options) == 0:
            raise ValueError(f"No options found for tag {tag}, sample {sample}, firstN {firstN}, testcut {testcut}.")
        elif len(options) > 1:
            print("Warning: multiple options for nboot found for reco:")
            nboot_options = []
            for i, opt in enumerate(options):
                m = re.search(r'_boot(\d+)', opt)
                if m:
                    nboot_options.append(int(m.group(1)))
                else:
                    nboot_options.append(0)
                print("\t%d: nboot=%d"%(i, nboot_options[-1]))
            print("Using the largest option. If this is not what you want, please specify nboot explicitly.")
            nboot = max(nboot_options)
        else:
            m = re.search(r'_boot(\d+)', options[0])
            if m:
                nboot = int(m.group(1))
            else:
                nboot = 0

    name = f'{tag}_{sample}'
    name += f'_boot{nboot}'
    if statN > 0:
        name += f'_{statN}stat{statK}'
    if firstN > 0:
        name += f'_first{firstN}'
    if testcut:
        name += '_TESTCUT'
    name += f'_{objsyst}_{wtsyst}'
    return name

def reco_folder(tag, sample, nboot, statN, statK, firstN, objsyst, wtsyst, testcut):
    name = reco_name(tag, sample, nboot, statN, statK, firstN, objsyst, wtsyst, testcut)

    path = os.path.join(
        datasets.basedir, tag, sample,
        'EECres4tee', 'CONSTRUCTED_RECO',
        name)
    return path
