import datasets
import re
import os
import numpy as np

def get_hist_paths(tag, sample, nboot, statN, statK, firstN, objsyst, wtsyst, what,
                   reweight=None, r123type='Philox', from_bkp=True, silent=False):
    basepath = os.path.join(datasets.basedir, tag, sample, 'EECres4tee')
    histpath_options = os.scandir(basepath)
    histpath_options = list(filter(
        lambda x: x.is_dir() and x.name.startswith(f'hists'),
        histpath_options
    ))
    if len(histpath_options) == 0:
        raise ValueError(f"No runs found for tag {tag}, sample {sample}.")
    elif len(histpath_options) > 1:
        sizes = []
        for histpath in histpath_options:
            m = re.search(r'file(\d+)to(\d+)', histpath.name)
            if m:
                start = int(m.group(1))
                end = int(m.group(2))
                sizes.append(end - start)
            else:
                raise ValueError(f"Cannot parse file range from {histpath}.")
        if not silent:
            print("Warning: multiple hist paths found:")
            for i,  in enumerate(sizes):
                print(f"\t{i}: {histpath_options[i].name} ({sizes[i]} files)")
            print("Using the largest option. There is currently no way to override this behavior")
        choice = np.argmax(sizes)
        histpath = histpath_options[choice]
    else:
        histpath = histpath_options[0]

    basepath = os.path.join(basepath, histpath.name) 

    if from_bkp:
        basepath = os.path.join(basepath, 'hists_bkp')
    else:
        basepath = os.path.join(basepath, objsyst)

    nominal_options = []
    extraboot_options = []
    total_nboot = 0

    for path in os.scandir(basepath):
        if path.is_file() and path.name.startswith('%s_%s_%s'%(what, objsyst, wtsyst)):
            if statN > 0 and '_%dstat%d' % (statN, statK) not in path.name:
                continue
            elif statN < 0 and 'stat' in path.name:
                continue

            if firstN > 0 and '_first%d' % firstN not in path.name:
                continue
            elif firstN < 0 and 'first' in path.name:
                continue

            if nboot > 0 and '_boot' in path.name and '_boot%d' % nboot not in path.name:
                continue
            elif nboot == 0 and '_boot' in path.name:
                continue

            if (reweight is not None) and (reweight not in path.name):
                continue

            if r123type is not None and r123type not in path.name:
                continue

            if 'skipNominal' in path.name:
                extraboot_options.append(path.path)
            else:
                nominal_options.append(path.path)

            if 'boot' in path.name:
                m = re.search(r'_boot(\d+)', path.name)
                if m:
                    this_nboot = int(m.group(1))
                    total_nboot += this_nboot
                else:
                    raise ValueError("Cannot parse nboot from %s." % path.name)

    if len(nominal_options) == 0:
        raise ValueError("No nominal options found for %s_%s_%s in %s." % (what, objsyst, wtsyst, basepath))
    elif len(nominal_options) > 1:
        if not silent:
            print("Warning: multiple nominal options found for %s_%s_%s in %s:" % (what, objsyst, wtsyst, basepath))
        nboots = []
        for i, opt in enumerate(nominal_options):
            if not silent:
                print("\t%d: %s" % (i, opt))
            m = re.search(r'_boot(\d+)', opt)
            if m:
                nboots.append(int(m.group(1)))
            else:
                nboots.append(0)
        if not silent:
            print("Using the largest option. If this is not what you want, please specify nboot explicitly.")
        choice = np.argmax(nboots)
        nominal_options = [nominal_options[choice]]


    return nominal_options[0], extraboot_options, total_nboot

def get_full_hist(tag, sample, nboot, statN, statK, firstN, objsyst, wtsyst, what, 
                  reweight=None, r123type=None, max_nboot=-1, 
                  from_bkp=True, silent=False):
    if nboot < 0:
        if not silent:
            print("passed nboot < 0: disambiguating...")

        nom, extraboot_options, _ = get_hist_paths(
            tag, sample, nboot, statN, statK, firstN, objsyst, wtsyst, what,
            reweight, r123type, from_bkp,
            silent=silent
        )
        if len(extraboot_options) > 0:
            nboot_options = {}
            for path in extraboot_options:
                m = re.search(r'_boot(\d+)', os.path.basename(path))
                if m:
                    nb = int(m.group(1))
                    if nb not in nboot_options:
                        nboot_options[nb] = 0
                    nboot_options[nb] += nb
                else:
                    raise ValueError("Cannot parse nboot from %s." % path)

            if len(nboot_options) == 0:
                nboot = 0
            elif len(nboot_options) == 1:
                nboot = list(nboot_options.keys())[0]
            else:
                if not silent:
                    print("Warning: multiple nboot per file options:")
                    for nb, count in nboot_options.items():
                        print("\t%d: %d samples" % (nb, count))
                    print("Chosing the option with the most samples")
                nboot = max(nboot_options, key=nboot_options.get)
        else:
            m = re.search(r'_boot(\d+)', nom)
            if m:
                nboot = int(m.group(1))
            else:
                nboot = 0

    nominal_path, extraboot_options, total_nboot = get_hist_paths(
        tag, sample, nboot, statN, statK, firstN, objsyst, wtsyst, what,
        reweight, r123type, from_bkp,
        silent=silent
    )

    import ioutil
    import hist
    from tqdm import tqdm

    Hnom = ioutil.wrapped_read_pickle(nominal_path, silent=silent)

    if max_nboot != 0 and len(extraboot_options) != 0:
        axes = []
        for ax in Hnom.axes:
            if ax.name == 'bootstrap':
                if max_nboot < 0:
                    actual_nboot = total_nboot
                elif max_nboot > total_nboot:
                    actual_nboot = total_nboot
                else:
                    #clip to nearest multiple of nboot
                    actual_nboot = nboot * (max_nboot // nboot)

                actual_nboot += ax.size
                axes.append(
                    hist.axis.Integer(0, actual_nboot, 
                                      overflow=False, underflow=False, 
                                      label='bootstrap', name='bootstrap')
                )
            else:
                axes.append(ax)
        
        Htot = hist.Hist(*axes, hist.storage.Double())
        Htot.view(flow=True)[:Hnom.axes['bootstrap'].size] += Hnom.view(flow=True)

        start = Hnom.axes['bootstrap'].size
        Nfiles_to_read = (actual_nboot - Hnom.axes['bootstrap'].size) // nboot
        files_to_read = extraboot_options[:Nfiles_to_read]
        if not silent:
            print("Reading extra bootstraps from %d files..." % len(files_to_read))

        for path in tqdm(files_to_read, disable=silent):
            if start >= actual_nboot:
                break
            Hnext = ioutil.wrapped_read_pickle(path, silent=True)
            Htot.view(flow=True)[start:start + nboot] += Hnext.view(flow=True)
            start += nboot
    
        return Htot
    else:
        return Hnom

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
    m = re.search(r'_SYST(?:-([a-zA-Z0-9_]+))*', name)
    if m:
        syst_l = m.group(0).split('-')[1:]
    else:
        syst_l = []

    m = re.search(r'_PROJECT(?:-([a-zA-Z0-9]+))*', name)
    if m:
        projectAxes = m.group(0).split('-')[1:]
    else:
        projectAxes=None

    smoothed = '_SMOOTHED' in name
    return tag, sample, nboot, statN, statK, firstN, syst_l, projectAxes, smoothed

def loss_name(tag, sample, nboot, statN, statK, firstN, syst_l, projectAxes, smoothed):
    if nboot < 0:
        options = os.listdir(os.path.join(datasets.basedir, tag, sample, 'EECres4tee', 'CONSTRUCTED_LOSSES'))
        options = list(filter(lambda x: x.startswith(f'{tag}_{sample}_'), options))

        if statN > 0:
            options = list(filter(lambda x: f'_{statN}stat{statK}' in x, options))
        else:
            options = list(filter(lambda x: 'stat' not in x, options))
        
        if firstN > 0:
            options = list(filter(lambda x: f'_first{firstN}' in x, options))
        else:
            options = list(filter(lambda x: '_first' not in x, options))

        if len(syst_l) > 0:
            syststr = '_SYST'
            for syst in syst_l:
                syststr += f'-{syst}'
            options = list(filter(lambda x: syststr in x, options))
        else:
            options = list(filter(lambda x: '_SYST' not in x, options))

        if projectAxes is not None and len(projectAxes) > 0:
            projectstr = '_PROJECT'
            for ax in projectAxes:
                projectstr += f'-{ax}'
            options = list(filter(lambda x: projectstr in x, options))
        else:
            options = list(filter(lambda x: '_PROJECT' not in x, options))

        if smoothed:
            options = list(filter(lambda x: '_SMOOTHED' in x, options))
        else:
            options = list(filter(lambda x: '_SMOOTHED' not in x, options))

        if len(options) == 0:
            raise ValueError(f"No options found for tag {tag}, sample {sample}, firstN {firstN}, syst_l {syst_l}, projectAxes {projectAxes}, smoothed {smoothed}.")
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
                print("\t\t", opt)
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
    if projectAxes is not None and len(projectAxes) > 0:
        name += '_PROJECT'
        for ax in projectAxes:
            name += f'-{ax}'
    if smoothed:
        name += '_SMOOTHED'
    if len(syst_l) > 0:
        name += '_SYST'
        for syst in syst_l:
            name += f'-{syst}'
    return name

def loss_folder(tag, sample, nboot, statN, statK, firstN, syst_l, projectAxes, smoothed):
    name = loss_name(tag, sample, nboot, statN, statK, firstN, syst_l, projectAxes, smoothed)

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

    m = re.search(r'_PROJECT(?:-([a-zA-Z0-9]+))*', name)
    if m:
        projectAxes = m.group(0).split('-')[1:]
    else:
        projectAxes = None
    return tag, sample, nboot, statN, statK, firstN, objsyst, wtsyst, projectAxes

def reco_name(tag, sample, nboot, statN, statK, firstN, objsyst, wtsyst, projectAxes):
    if nboot < 0:
        options = os.listdir(os.path.join(datasets.basedir, tag, sample, 'EECres4tee', 'CONSTRUCTED_RECO')) 
        options = list(filter(lambda x: x.startswith(f'{tag}_{sample}_'), options))

        if statN > 0:
            options = list(filter(lambda x: f'_{statN}stat{statK}' in x, options))
        else:
            options = list(filter(lambda x: 'stat' not in x, options))

        if firstN > 0:
            options = list(filter(lambda x: f'_first{firstN}' in x, options))
        else:
            options = list(filter(lambda x: '_first' not in x, options))

        if projectAxes is not None and len(projectAxes) > 0:
            projectstr = '_PROJECT'
            for ax in projectAxes:
                projectstr += f'-{ax}'
            options = list(filter(lambda x: projectstr in x, options))
        else:
            options = list(filter(lambda x: '_PROJECT' not in x, options))

        if len(options) == 0:
            raise ValueError(f"No options found for tag {tag}, sample {sample}, firstN {firstN}, projectAxes {projectAxes}.")
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
    if projectAxes is not None and len(projectAxes) > 0:
        name += '_PROJECT'
        for ax in projectAxes:
            name += f'-{ax}'
    name += f'_{objsyst}_{wtsyst}'
    return name

def reco_folder(tag, sample, nboot, statN, statK, firstN, objsyst, wtsyst, projectAxes):
    name = reco_name(tag, sample, nboot, statN, statK, firstN, objsyst, wtsyst, projectAxes)

    path = os.path.join(
        datasets.basedir, tag, sample,
        'EECres4tee', 'CONSTRUCTED_RECO',
        name)
    return path
