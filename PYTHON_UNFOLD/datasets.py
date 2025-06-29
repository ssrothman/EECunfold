import pyarrow.dataset as ds
import re
import hist
import pickle
import os.path
import os

basedir = '/ceph/submit/data/group/cms/store/user/srothman/EEC/'
unf_basedir = '/home/submit/srothman/work/EEC/EECunfold/data'

def try_to_read_pkl(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        import sys
        sys.stderr.write(f"\n\nError reading pickle file {path}: {e}\n\n")
        raise e

def check_skimpath(subpath):
    if not subpath.is_dir():
        return False

    if not subpath.name.startswith('hists'):
        return False

    return True

def check_histpath(subpath, whichobj, objsyst, wtsyst, 
                   skipNominal, statN, statK,
                   firstN,
                   reweight, boot_per_file):

    if not subpath.is_file():
        return False

    if not subpath.name.startswith("%s_%s_%s"%(whichobj, objsyst, wtsyst)):
        return False

    if skipNominal is not None:
        if skipNominal and 'skipNominal' not in subpath.name:
            return False
        if not skipNominal and 'skipNominal' in subpath.name:
            return False
    
    if statN > 0 and "%dstat%d"%(statN, statK) not in subpath.name:
        return False
    if statN <= 0 and 'stat' in subpath.name:
        return False

    if reweight is not None and reweight not in subpath.name:
        return False

    if boot_per_file > 0 and 'boot' in subpath.name and 'boot%d' % boot_per_file not in subpath.name:
        return False
    if boot_per_file == 0 and 'boot' in subpath.name:
        return False

    if firstN > 0 and 'first%d' % firstN not in subpath.name:
        return False
    if firstN <= 0 and 'first' in subpath.name:
        return False

    if os.stat(subpath).st_size == 0:
        #guard with lots of newlines to protect from tqdm
        import sys
        sys.stderr.write(f"\n\nWarning: {subpath.name} is empty, skipping\n\n")
        return False 

    return True

def maybe_chose_option(options, user_input=True):
    if len(options) == 0:
        raise RuntimeError("No options found")
    elif len(options) == 1:
        return options[0]
    else:
        if user_input:
            print()
            print("Multiple options found:")
            for i, option in enumerate(options):
                print(f"{i}: {option}")
            print()
            choice = int(input("Please select a number: "))
            return options[choice]
            print()
        else:
            import re
            import numpy as np
            nboots = []
            for option in options:
                m = re.search(r'_boot(\d+)', option)
                if m:
                    nboots.append(int(m.group(1)))
                else:
                    nboots.append(0)
            best = np.argmax(nboots)
            return options[best]
        
def get_dataset(runtag, tag, skimmer, objsyst, whichobj):
    thepath = os.path.join(basedir, runtag, tag, skimmer)

    subpaths = os.scandir(thepath)
    options = []
    for subpath in subpaths:
        if not (subpath.is_dir()):
            continue
        options.append(subpath.name)

    if (len(options) == 1):
        #print()
        #print("Only one option found: %s" % options[0])
        #print("No user input needed :D")
        #print()
        thepath = os.path.join(thepath, options[0])
    else:
        print()
        print("Multiple options found:")
        for i, option in enumerate(options):
            print(f"{i}: {option}")
        print()
        choice = int(input("Please select a number: "))
        thepath = os.path.join(thepath, options[choice])
        print()

    thepath = os.path.join(thepath, objsyst, whichobj)
    #print("The path is: %s" % thepath)

    return ds.dataset(thepath, format="parquet")

def get_counts(runtag, tag):
    thepath = os.path.join(basedir, runtag, tag, 'Count')

    subpaths = os.scandir(thepath)
    options = []
    for subpath in subpaths:
        options.append(subpath.name)

    if (len(options) == 1):
        #print()
        #print("Only one option found: %s" % options[0])
        #print("No user input needed :D")
        #print()
        thepath = os.path.join(thepath, options[0])
    else:
        print()
        print("Multiple options found:")
        for i, option in enumerate(options):
            print(f"{i}: {option}")
        print()
        choice = int(input("Please select a number: "))
        thepath = os.path.join(thepath, options[choice])
        print()

    #print("The path is: %s" % thepath)
    return try_to_read_pkl(thepath)['num_evt']

def get_pickled_histogram(runtag, tag, skimmer,
                          objsyst, wtsyst, whichobj,
                          statN, statK, 
                          boot_per_file, firstN,
                          reweight,
                          max_nboot,
                          verbose=False):

    if type(boot_per_file) not in [list, tuple]:
        boot_per_file = [boot_per_file]

    thepath = os.path.join(basedir, runtag, tag, skimmer)

    subpaths = os.scandir(thepath)
    options = []
    for subpath in subpaths:
        if check_skimpath(subpath):
            options.append(subpath.name)

    thepath = os.path.join(thepath,
                           maybe_chose_option(options, user_input=False),
                           objsyst)

    subpaths = os.scandir(thepath)
    options = []
    for subpath in subpaths:
        if check_histpath(subpath, whichobj, objsyst, wtsyst, 
                          skipNominal=False, 
                          statN=statN, statK=statK, 
                          firstN=firstN,
                          reweight=reweight, boot_per_file=-1):
            options.append(subpath.name)

    nomfile = os.path.join(thepath, maybe_chose_option(options, user_input=False))

    H = try_to_read_pkl(nomfile)

    if type(H) in [list, tuple]:
        H = H[0]

    #find additional bootstraps

    subpaths = list(os.scandir(thepath))
    #first chose which boot_per_file to use
    lens= {boot : 0 for boot in boot_per_file}
    for bootcheck in boot_per_file:
        for subpath in subpaths:
            if check_histpath(subpath, whichobj, objsyst, wtsyst, 
                              skipNominal=True,
                              statN=statN, statK=statK, firstN=firstN,
                              reweight=reweight, boot_per_file=bootcheck):
                lens[bootcheck] += bootcheck

    best_boot_per_file = max(lens, key=lens.get)

    Hboots = []
    Nboot_so_far = H.axes['bootstrap'].size - 1
    for subpath in subpaths:
        if check_histpath(subpath, whichobj, objsyst, wtsyst,
                          skipNominal=True,
                          statN=statN, statK=statK, firstN=firstN,
                          reweight=reweight, boot_per_file=best_boot_per_file):
            if verbose:
                print("Found bootstrap file: %s" % subpath.name)

            Hnext = try_to_read_pkl(os.path.join(thepath, subpath.name))
            if type(Hnext) in [list, tuple]:
                Hnext = Hnext[0]

            Nboot_so_far += Hnext.axes['bootstrap'].size
            Hboots.append(Hnext)
            if max_nboot >= 0 and Nboot_so_far >= max_nboot:
                break

    if len(Hboots) > 0:
        totalboot = H.axes['bootstrap'].size
        for Hb in Hboots:
            totalboot += Hb.axes['bootstrap'].size

        target_axes = [hist.axis.Integer(0, totalboot, 
                                         overflow=False, underflow=False,
                                         label='bootstrap', name='bootstrap')]
        for ax in H.axes:
            if ax.name == 'bootstrap':
                continue
            else:
                target_axes.append(ax)
        Htot = hist.Hist(
            *target_axes,
            hist.storage.Double()
        )
        Htot.view(flow=True)[:H.axes['bootstrap'].size] += H.view(flow=True)
        offset = H.axes['bootstrap'].size
        for Hb in Hboots:
            nextsize = Hb.axes['bootstrap'].size
            Htot.view(flow=True)[offset:offset+nextsize] += Hb.view(flow=True)
            offset += nextsize
        H = Htot

    return H

def get_pickled_histogram_sum(tags, xsecs, runtag, skimmer, 
                              objsyst, wtsyst, whichobj,
                              statN, statK, firstN,
                              boot_per_file, 
                              reweight,
                              max_nboot):
    if type(xsecs) not in [list, tuple]:
        xsecs = [xsecs] * len(tags)
    elif len(xsecs) == 1:
        xsecs = xsecs * len(tags)

    if len(tags) != len(xsecs):
        raise ValueError("tags and xsecs must have the same length, or xsecs must have length 1")

    H = None
    for tag, xsec in zip(tags, xsecs):
        print("Reading %s (xsec = %g)"%(tag, xsec))
        Hnext = get_pickled_histogram(runtag, tag, skimmer, 
                                      objsyst, wtsyst, whichobj,
                                      statN, statK, firstN=firstN,
                                      boot_per_file=boot_per_file,
                                      reweight=reweight,
                                      max_nboot=max_nboot,
                                      verbose=False)
        numevt = get_counts(runtag, tag)
        samplewt = xsec/numevt * 1000
        print("\tFound H with %d bootstraps" % (Hnext.axes['bootstrap'].size - 1))

        if H is None:
            H = (Hnext * samplewt)
        elif 'bootstrap' in Hnext.axes.name:
            if Hnext.axes['bootstrap'].size < H.axes['bootstrap'].size:
                H = H[{'bootstrap' : slice(None, Hnext.axes['bootstrap'].size)}]
                H += (Hnext * samplewt)
            elif Hnext.axes['bootstrap'].size > H.axes['bootstrap'].size:
                Hnext = Hnext[{'bootstrap' : slice(None, H.axes['bootstrap'].size)}]
                H += (Hnext * samplewt)
            else:
                H += (Hnext * samplewt)
        else:
            H += (Hnext * samplewt)
    return H
