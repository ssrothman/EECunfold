import pyarrow.dataset as ds
import re
import hist
import pickle
import os.path

basedir = '/ceph/submit/data/group/cms/store/user/srothman/EEC/'
unf_basedir = '/home/submit/srothman/work/EEC/EECunfold/data'

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

    with open(thepath, 'rb') as f:
        return pickle.load(f)['num_evt']

def get_unfolded_histogram(name):
    with open(os.path.join(unf_basedir, name+'.pkl'), 'rb') as f:
        return pickle.load(f)

def get_pickled_histogram(runtag, tag, skimmer, objsyst, wtsyst, whichobj,
                          statN=-1, statK=-1, shuffle_boots=False,
                          boot_per_file=-1,
                          reweight='kinreweight',
                          max_nboot=-1):
    thepath = os.path.join(basedir, runtag, tag, skimmer)

    subpaths = os.scandir(thepath)
    options = []
    for subpath in subpaths:
        if not (subpath.is_dir()):
            continue
        if not (subpath.name.startswith('hists')):
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

    thepath = os.path.join(thepath, objsyst)
    subpaths = os.scandir(thepath)
    options = []
    for subpath in subpaths:
        if subpath.is_dir():
            continue
        if subpath.name.startswith("%s_%s_%s"%(whichobj, objsyst, wtsyst)) and 'skipNominal' not in subpath.name:
            if statN > 0 and "%dstat%d"%(statN, statK) not in subpath.name:
                continue
            if statN <=0 and 'stat' in subpath.name:
                continue 
            if reweight is not None and reweight not in subpath.name:
                continue

            if boot_per_file >0 and 'boot' in subpath.name and 'boot%d' % boot_per_file not in subpath.name:
                print("couldn't find boot%d in %s"%(boot_per_file, subpath.name))
                continue

            options.append(subpath.name)

    if (len(options) == 1):
        #print()
        #print("Only one option found: %s" % options[0])
        #print("No user input needed :D")
        #print()
        thefile = os.path.join(thepath, options[0])
        choice=0
    elif len(options) == 0:
        print("Uh oh no options")
        print("Looking for %s + %s"%(thepath, '%s_%s_%s'%(whichobj, objsyst, wtsyst)))
        print(runtag, tag, skimmer, objsyst, wtsyst, whichobj)
        raise RuntimeError("No options found for %s + %s"%(thepath, '%s_%s_%s'%(whichobj, objsyst, wtsyst)))
        print()
    else:
        print()
        print("Multiple options found:")
        for i, option in enumerate(options):
            print(f"{i}: {option}")
        print()
        choice = int(input("Please select a number: "))
        thefile = os.path.join(thepath, options[choice])
        print()

    with open(thefile, 'rb') as f:
        H = pickle.load(f)

    if type(H) in [list, tuple]:
        H = H[0]

    #find additional bootstraps
    Hboots = []

    if 'first' in options[choice]:
        firstN = int(re.search(r'first(\d+)', options[choice]).group(1))
    else:
        firstN = -1

    used_rngs = []
    if 'rng' in options[choice]:
        used_rngs.append(int(re.search(r'rng(\d+)', options[choice]).group(1)))

    subpaths = list(os.scandir(thepath))
    if shuffle_boots:
        np.random.shuffle(subpaths)

    Nboot_so_far = H.axes['bootstrap'].size - 1
    for subpath in subpaths:
        if subpath.is_dir():
            continue
        if not subpath.name.startswith('%s_%s_%s'%(whichobj, objsyst, wtsyst)):
            continue
        if firstN > 0:
            if not 'first%d'%firstN in subpath.name:
                continue
        else:
            if 'first' in subpath.name:
                continue
        if 'boot' not in subpath.name:
            continue
        if reweight is not None and reweight not in subpath.name:
            continue

        if statN > 0 and "%dstat%d"%(statN, statK) not in subpath.name:
            continue

        if statN <=0 and 'stat' in subpath.name:
            continue
        
        if boot_per_file >0 and 'boot%d' % boot_per_file not in subpath.name:
            continue

        nextrng = int(re.search(r'rng(\d+)', subpath.name).group(1))
        if nextrng in used_rngs:
            continue

        used_rngs.append(nextrng)
        with open(os.path.join(thepath, subpath.name), 'rb') as f:
            print(os.path.join(thepath, subpath.name))
            Hnext = pickle.load(f)
            if type(Hnext) in [list, tuple]:
                Hnext = Hnext[0]
            if 'skipNominal' not in subpath.name:
                Hnext = Hnext[{'bootstrap' : slice(1, None)}]

        Nboot_so_far += Hnext.axes['bootstrap'].size
        Hboots.append(Hnext)
        if max_nboot >= 0 and Nboot_so_far >= max_nboot:
            break

    expected_sumwt = H.sum(flow=True)
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
            expected_sumwt += Hb.sum(flow=True)
        H = Htot

    print("Total weight = %g (expected %g)"%(H.sum(flow=True), expected_sumwt))
    return H

def get_pickled_histogram_sum(tags, xsecs, runtag, skimmer, 
                              objsyst, wtsyst, whichobj,
                              statN, statK):
    H = None
    for tag, xsec in zip(tags, xsecs):
        Hnext = get_pickled_histogram(runtag, tag, skimmer, 
                                      objsyst, wtsyst, whichobj,
                                      statN, statK)
        numevt = get_counts(runtag, tag)
        samplewt = xsec/numevt * 1000

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
