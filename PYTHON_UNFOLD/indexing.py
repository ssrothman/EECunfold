import numpy as np
import torch
import itertools
import json
import os

class Index:
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return f"Index({self.val})"

    def __str__(self):
        return f"Index({self.val})"

'''
Not part of the public interface
'''
class BinningBlock:
    def __init__(self):
        self.strides = []
        self.extents = []
        self.axis_names = []
        self.ax_details = {}
        self.Nax = 0
        self.total_size = 1
        self.offset = 0

    def __eq__(self, other):
        if not isinstance(other, BinningBlock):
            return False

        if self.Nax != other.Nax:
            return False

        if self.total_size != other.total_size:
            return False

        if self.axis_names != other.axis_names:
            return False

        for i, ax_name in enumerate(self.axis_names):
            if ax_name not in other.axis_names:
                return False

            if self.extents[i] != other.extents[other.axis_names.index(ax_name)]:
                return False

            if self.ax_details[ax_name]['edges'] != other.ax_details[ax_name]['edges']:
                return False

        return True

    def copy(self):
        result = BinningBlock()
        result.from_dict(self.to_dict())
        return result

    def to_dict(self):
        result = {
            'axis_names': self.axis_names,
            'Nax': self.Nax,
            'extents': self.extents,
            'ax_details': self.ax_details,
            'strides': self.strides,
            'total_size': self.total_size,
            'offset': self.offset
        }
        return result

    def from_dict(self, thedict):
        self.axis_names = thedict['axis_names']
        self.Nax = thedict['Nax']
        self.extents = thedict['extents']
        self.ax_details = thedict['ax_details']
        self.strides = thedict['strides']
        self.total_size = thedict['total_size']
        self.offset = thedict['offset']

    def from_hist(self, H):
        self.Nax = len(H.axes)

        for ax in H.axes:
            self.axis_names.append(ax.name)
            self.extents.append(ax.extent)
            edges = ax.edges.tolist()
            if ax.traits.underflow:
                edges = [-np.inf] + edges
            if ax.traits.overflow:
                edges = edges + [np.inf]
            self.ax_details[ax.name] = {
                'edges': edges,
                'extent': ax.extent,
                'minedge': edges[0],
                'maxedge': edges[-1]
            }
            self.total_size *= ax.extent

        self.offset = 0

        self.calculate_strides()

    def calculate_strides(self):
        self.strides = [0] * self.Nax
        self.strides[self.Nax - 1] = 1
        for i in range(self.Nax - 1, 0, -1):
            self.strides[i-1] = self.strides[i] * self.extents[i]

    def rebin(self, rebinning_spec):
        if type(rebinning_spec) is str:
            with open(rebinning_spec, 'r') as f:
                rebinning_spec = json.load(f)
        elif type(rebinning_spec) is not dict:
            raise ValueError("Rebinning specification must be a dictionary or a path to a JSON file.")

        newblocks = []
        running_offset = 0
        for specblock in rebinning_spec['spec']:
            nextblock = BinningBlock()
            nextblock.axis_names = self.axis_names.copy()
            nextblock.Nax = self.Nax
            for name in self.axis_names:
                extent = len(specblock[name]) - 1
                nextblock.extents.append(extent)
                edges = [self.ax_details[name]['edges'][i] for i in specblock[name]]
                nextblock.ax_details[name] = {
                    'edges' : edges,
                    'extent' : extent,
                    'minedge' : edges[0],
                    'maxedge' : edges[-1]
                }
                nextblock.total_size *= extent

            nextblock.calculate_strides()
            nextblock.offset = running_offset
            running_offset += nextblock.total_size

            newblocks.append(nextblock)

        return newblocks

    def edge_to_index(self, name, edge):
        if edge is None: #special case, needed for slices
            return None

        if type(edge) is Index:
            return edge.val

        edges = self.ax_details[name]['edges']
        for i, e in enumerate(edges):
            if e == edge:
                return i
            if np.abs(e-edge) < 1e-5:
                return i

        print()
        print(edge)
        print(self.ax_details[name]['edges'])
        print()
        raise ValueError(f"Edge {edge} not found in axis {name} edges.")

    def edges_to_indices(self, name, edges):
        if type(edges) in [int, float, Index]:
            return self.edge_to_index(name, edges)
        elif type(edges) is list:
            return [self.edge_to_index(name, edge) for edge in edges]
        elif type(edges) is tuple:
            return tuple(self.edge_to_index(name, edge) for edge in edges)
        elif type(edges) is slice:
            start = self.edge_to_index(name, edges.start)
            stop = self.edge_to_index(name, edges.stop)
            return slice(start, stop, edges.step)
        elif type(edges) is dict:
            return {name: self.edges_to_indices(name, edges[name]) for name in edges}
        else:
            raise ValueError(f"Invalid type for edges: {type(edges)}. Expected int, list, or slice.")

    def index_to_edge(self, name, index):
        if index is None: #special case, needed for slices
            return None

        if index < 0 or index >= self.ax_details[name]['extent']:
            raise IndexError(f"Index {index} out of bounds for axis {name}.")

        return self.ax_details[name]['edges'][index]

    def indices_to_edges(self, name, indices):
        if type(indices) is int:
            return self.index_to_edge(name, indices)
        elif type(indices) is list:
            return [self.index_to_edge(name, idx) for idx in indices]
        elif type(indices) is tuple:
            return tuple(self.index_to_edge(name, idx) for idx in indices)
        elif type(indices) is dict:
            return {name: self.indices_to_edges(name, indices[name]) for name in indices}
        elif type(indices) is slice:
            return slice(self.index_to_edge(name, indices.start),
                         self.index_to_edge(name, indices.stop),
                         indices.step)
        else:
            raise ValueError(f"Invalid type for indices: {type(indices)}. Expected int, list, or slice.")

    def flatten_index(self, **theindices):
        for name in theindices:
            if name not in self.axis_names:
                raise ValueError(f"Invalid axis name: {name}")

        for name in self.axis_names:
            if name not in theindices:
                raise ValueError(f"Missing value for axis: {name}")

        indices = []
        for name in self.axis_names:
            indices.append(theindices[name])

        idx = 0
        for i, index in enumerate(indices):
            if index < 0 or index >= self.extents[i]:
                raise IndexError(f"Index {index} out of bounds for axis {self.axis_names[i]}")
            idx += index * self.strides[i]

        return idx

    def unflatten_index(self, index):
        result = {}
        for i, name in enumerate(self.axis_names):
            result[name] = (index // self.strides[i]) % self.extents[i]
        return result

    def get_slice_indices(self, **sliceindices): 
        for name in sliceindices:
            if name not in self.axis_names:
                raise ValueError(f"Invalid axis name: {name}")
            if type(sliceindices[name]) not in [list, tuple]:
                raise ValueError(f"Slice indices for axis {name} must be a list or tuple.")
            if len(sliceindices[name]) != 2:
                raise ValueError(f"Slice indices for axis {name} must contain exactly two elements: (start, stop).")

        indices = []
        for i in range(self.total_size):
            index = self.unflatten_index(i)
            accepted = True
            for axname in sliceindices:
                if index[axname] < sliceindices[axname][0] or index[axname] >= sliceindices[axname][1]:
                    accepted = False
                    break
            if accepted:
                indices.append(i)

        return np.asarray(indices, dtype=np.int64)

    def get_slice_from_edges(self, data, **edges):
        return self.get_slice_from_indices(data, **{name: self.edges_to_indices(name, edges[name]) for name in edges})

    def get_slice_from_indices(self, data, **indices):
        indices = self.get_slice_indices(**indices)
        if type(data) is torch.Tensor:
            return torch.take(data, torch.tensor(self.offset + indices, device=data.device))
        else:
            return np.take(data, self.offset+indices, axis=0)

    def assign_to_indices(self, data, values, **indices):
        indices = self.get_slice_indices(**indices)
        data[indices + self.offset] = values

    def assign_to_indices_2d(self, data, values, **indices):
        indices = self.get_slice_indices(**indices)
        indexmin = np.min(indices)
        indexmax = np.max(indices)
        indexrange = indexmax - indexmin +1
        if indexrange != values.shape[0]:
            raise ValueError("Can only assign to a continuous block of indices")
        indexing = slice(self.offset + indexmin, 
                         self.offset + indexmax + 1)
        print("DATA SHAPE:", data.shape)
        print("VALUES SHAPE:", values.shape)
        print("INDICES SHAPE:", indices.shape)
        print("INDEXED DATA SHAPE", data[indexing, indexing].shape)

        data[indexing, indexing] = values

    def project_out(self, data, axis_name):
        blockdata = data[self.offset: self.offset + self.total_size]
        shape = []
        for i, name in enumerate(self.axis_names):
            shape.append(self.ax_details[name]['extent'])
            if name == axis_name:
                whichax = i
        extradims = list(data.shape[1:])
        shape = shape + extradims

        blockdata = blockdata.reshape(shape)
        result = np.sum(blockdata, axis=whichax, keepdims=False)
        result = result.reshape((-1, *extradims))

        newblock = BinningBlock()
        newblock.axis_names = self.axis_names.copy()
        newblock.axis_names.pop(whichax)
        newblock.Nax = self.Nax - 1
        newblock.extents = self.extents.copy()
        newblock.extents.pop(whichax)
        newblock.ax_details = self.ax_details.copy()
        newblock.ax_details.pop(axis_name)
        newblock.total_size = int(np.prod(newblock.extents))
        newblock.calculate_strides()
        newblock.offset = None

        return result, newblock

    def value_at(self, data, **indices):
        return data[self.offset+self.flatten_index(**indices)]

    def edges_in_block(self, **theedges):
        for name in theedges:
            if name not in self.axis_names:
                raise ValueError(f"Invalid axis name: {name}")

        for name in theedges:
            edges = theedges[name]
            allowedmin = self.ax_details[name]['minedge']
            allowedmax = self.ax_details[name]['maxedge']
            if type(edges) not in [list,tuple]:
                raise ValueError(f"Edges for axis {name} must be a list or tuple.")
            elif len(edges) != 2:
                raise ValueError(f"Edges for axis {name} must contain exactly two elements: (min, max).")

            if type(edges[0]) in [int, float]:
                if edges[0] < allowedmin:
                    return False
            elif type(edges[0]) is Index:
                if edges[0].val < 0 or edges[0].val > self.ax_details[name]['extent']:
                    return False
            else:
                raise ValueError(f"Invalid type for edge {edges[0]} on axis {name}. Expected int, float, or Index.")

            if type(edges[1]) in [int, float]:
                if edges[1] > allowedmax:
                    return False
            elif type(edges[1]) is Index:
                if edges[1].val < 0 or edges[1].val > self.ax_details[name]['extent']:
                    return False
            else:
                raise ValueError(f"Invalid type for edge {edges[1]} on axis {name}. Expected int, float, or Index.")

        return True

    def clip_edges_to_block(self, **theedges):
        for name in theedges:
            if name not in self.axis_names:
                raise ValueError(f"Invalid axis name: {name}")

        clippededges = {}
        for name in theedges:
            edges = theedges[name]
            allowedmin = self.ax_details[name]['minedge']
            allowedmax = self.ax_details[name]['maxedge']
            if type(edges) not in [list, tuple]:
                raise ValueError(f"Edges for axis {name} must be a list or tuple.")
            elif len(edges) != 2:
                raise ValueError(f"Edges for axis {name} must contain exactly two elements: (min, max).")

            if type(edges[0]) in [int, float]:
                clippedmin = max(edges[0], allowedmin)
            elif type(edges[0]) is Index:
                clippedmin = edges[0]
            else:
                raise ValueError(f"Invalid type for edge {edges[0]} on axis {name}. Expected int, float, or Index.")

            if type(edges[1]) in [int, float]:
                clippedmax = min(edges[1], allowedmax)
            elif type(edges[1]) is Index:
                clippedmax = edges[1]
            else:
                raise ValueError(f"Invalid type for edge {edges[1]} on axis {name}. Expected int, float, or Index.")

            if type(clippedmin) in [int, float] and type(clippedmax) in [int, float] and clippedmax <= clippedmin:
                return None

            clippededges[name] = (clippedmin, clippedmax)

        return clippededges

class Binning:
    def __init__(self):
        self.blocks = []
        self.axis_names = []
        self.Nax = 0

    def copy(self):
        result = Binning()
        result.from_dict(self.to_dict())
        return result

    '''
    Initialize from a hist.Hist object
    '''
    def setup_from_histogram(self, H):
        self.Nax = len(H.axes)
        block = BinningBlock()
        block.from_hist(H)
        self.blocks = [block]
        self.axis_names = block.axis_names

    '''
    Write to json
    '''
    def dump_to_file(self, file):
        print("Writing binning spec to file: %s" % file)
        os.makedirs(os.path.dirname(file), exist_ok=True)
        resultdict = self.to_dict()
        with open(file, 'w') as f:
            json.dump(resultdict, f, indent=4)

    '''
    Read from json
    '''
    def load_from_file(self, file):
        print("Reading binning spec from file: %s" % file)

        with open(file, 'r') as f:
            resultdict = json.load(f)

        self.from_dict(resultdict)

    '''
    Dump to python dictionary
    '''
    def to_dict(self):
        resultdict = {}
        resultdict['axis_names'] = self.axis_names
        resultdict['Nax'] = self.Nax
        resultdict['blocks'] = []
        for block in self.blocks:
            resultdict['blocks'].append(block.to_dict())
        return resultdict

    '''
    Initialize from python dictionary
    '''
    def from_dict(self, resultdict):
        self.axis_names = resultdict['axis_names']
        self.Nax = resultdict['Nax']
        self.blocks = []
        for blockdata in resultdict['blocks']:
            block = BinningBlock()
            block.from_dict(blockdata)
            self.blocks.append(block)
        
    '''
    Lookup value in a specific BIN index
    '''
    def value_at(self, data, **theindices):
        in_block = np.zeros(len(self.blocks), dtype=bool)
        for i, block in enumerate(self.blocks):
            if block.indices_in_block(**theindices):
                in_block[i] = True

        if np.sum(in_block) == 0:
            raise ValueError("No block contains the specified indices.")
        elif np.sum(in_block) > 1:
            print("Multiple blocks contain the specified indices.")
            print("Indices:")
            for name in theindices:
                print("\t",name, ':', theindices[name])
            print("Blocks:", np.where(in_block)[0])
            raise ValueError("Multiple blocks contain the specified indices.")
        else:
            whichblock = np.argmax(in_block)
            return self.blocks[whichblock].value_at(data, **theindices)

    '''
    Get a particular slice
    theedges must be a dictionary with elements
        key: (low, high)
    where key is in self.axis_names
    and (low, high) are bin edges for that axis

    Not all axes need to be specified

    NB only indexes along first axis. If you have more axes (eg bootstrapping),
        these will be ignored as long as they are not axis 0
    This is a useful hack, as slicing can be slow for really big arrays
    '''
    def get_slice(self, data, **theedges):
        overlap_block = np.ones(len(self.blocks), dtype=bool)
        clipped_edges = []
        for i, block in enumerate(self.blocks):
            clipped = block.clip_edges_to_block(**theedges)
            if clipped is None:
                overlap_block[i] = False
                clipped_edges.append(None)
            else:
                clipped_edges.append(clipped)
        if np.sum(overlap_block) == 0:
            print("No blocks overlap with the specified edges.")

        if type(data) is torch.Tensor:
            result = torch.empty((0, *data.shape[1:]), dtype=data.dtype, device=data.device)
        else:
            result = np.empty((0, *data.shape[1:]), dtype=data.dtype)

        for i, block in enumerate(self.blocks):
            if overlap_block[i]:
                theslice = block.get_slice_from_edges(data, **clipped_edges[i])
                if type(data) is torch.Tensor:
                    result = torch.cat((result, theslice), dim=0)
                else:
                    result = np.append(result, theslice, axis=0)
        return result

    def get_slice_cov2d(self, data, **theedges):
        result = self.get_slice(data.T, **theedges)
        return self.get_slice(result.T, **theedges)

    def get_continuous_slice(self, data, **theedges):
        in_block = np.empty(len(self.blocks), dtype=bool)
        for i, block in enumerate(self.blocks):
            in_block[i] = block.edges_in_block(**theedges)

        if np.sum(in_block) == 0:
            raise ValueError("No block contains the specified edges.")
        elif np.sum(in_block) > 1:
            raise ValueError("Multiple blocks contain the specified edges.")

        whichblock = np.argmax(in_block)
        block = self.blocks[whichblock]
        return block.get_slice_from_edges(data, **theedges), block


    '''
    Project out an axis from the data

    Makes some pretty strong assumptions about the order of the blocks 
    wrt the axis you are projecting out. Caution!!
    '''
    def project_out(self, data, axis_name):
        startsum = data.sum(axis=None)

        result = np.empty((0, *data.shape[1:]), dtype=data.dtype)
        newbinning = Binning()
        newbinning.axis_names = self.axis_names.copy()
        newbinning.axis_names.remove(axis_name)
        newbinning.Nax = self.Nax - 1

        running_offset = 0
        prevblock = None
        for block in self.blocks:
            blockdata, newblock = block.project_out(data, axis_name)

            if prevblock is not None and newblock == prevblock:
                result[-newblock.total_size:] += blockdata
            else:
                result = np.append(result, blockdata, axis=0)
                newblock.offset = running_offset
                running_offset += newblock.total_size
                newbinning.blocks.append(newblock)
    
            prevblock = newblock

        endsum = result.sum(axis=None)
        if not np.isclose(startsum, endsum):
            raise ValueError(f"Projection changed the sum of the data: {startsum} -> {endsum}")

        return result, newbinning

    def project_out_cov2d(self, data, axis_name):
        result, newbinning = self.project_out(data.T, axis_name)
        result, _ = self.project_out(result.T, axis_name)
        return result, newbinning

    '''
    Rebin data according to a supplied spec
    An example spec is given in rebinnings/example.json

    Returns (rebinned data, rebinned Binning() instance for interacting with the data)
    '''
    def rebin(self, data, rebinning_spec):
        if len(self.blocks) != 1:
            raise ValueError("Can only rebin single-block binnings")

        if type(rebinning_spec) is str:
            with open(rebinning_spec, 'r') as f:
                rebinning_spec = json.load(f)
        elif type(rebinning_spec) is not dict:
            raise ValueError("Rebinning specification must be a dictionary or a path to a JSON file.")

        startsum = data.sum(axis=None)

        #check consistency
        for i, specblock in enumerate(rebinning_spec['spec']):
            for name in self.axis_names:
                if name not in specblock:
                    raise ValueError(f"Axis {name} not found in rebinning specification.")
            for name in specblock:
                if name not in self.axis_names:
                    raise ValueError(f"Axis {name} not found in histogram axes.")
                if type(specblock[name]) is not list:
                    raise ValueError(f"Axis {name} in rebinning specification must be a list of indices.")
                if np.min(specblock[name]) < 0:
                    raise ValueError(f"Axis {name} in rebinning specification contains negative indices.")

        extradims = list(data.shape[1:])
        result = np.empty((0, *extradims))
        for specblock in rebinning_spec['spec']:
            #force the index values into the right order
            #and ensure indices are sorted
            nextvals = self.get_specblock_binning(data, specblock)
            nextvals = nextvals.reshape((-1, *extradims))
            result = np.append(result, nextvals, axis=0)

        newbinning = Binning()
        newbinning.blocks = self.blocks[0].rebin(rebinning_spec)
        newbinning.axis_names = self.axis_names
        newbinning.Nax = self.Nax
            
        endsum = result.sum(axis=None)

        if not np.isclose(startsum, endsum):
            raise ValueError(f"Rebinning changed the sum of the data: {startsum} -> {endsum}")

        return result, newbinning

    def rebin_cov2d(self, data, rebinning_spec):
        result, newbinning = self.rebin(data.T, rebinning_spec)
        result, _ = self.rebin(result.T, rebinning_spec)
        return result, newbinning

    def get_blocks(self, axes):
        global_edges = {name : [] for name in self.axis_names}
        for name in self.axis_names:
            for block in self.blocks:
                global_edges[name] += block.ax_details[name]['edges']

        for name in self.axis_names:
            global_edges[name] = sorted(list(set(global_edges[name])))

        ranges = [range(len(global_edges[name]) - 1) for name in axes]
        import itertools
        result = []
        for indices in itertools.product(*ranges):
            theslice = {}
            for i, name in enumerate(axes):
                theslice[name] = (global_edges[name][indices[i]], 
                                  global_edges[name][indices[i] + 1])

            globalindices = np.empty((0,), dtype=np.int64)
            for block in self.blocks:
                blockslice = block.clip_edges_to_block(**theslice)
                if blockslice is not None:
                    blockslice = {name: block.edges_to_indices(name, blockslice[name]) for name in blockslice}
                    nextindices = block.offset+block.get_slice_indices(**blockslice)
                    globalindices = np.append(globalindices, nextindices)

            minindex = np.min(globalindices)
            maxindex = np.max(globalindices)
            size = maxindex - minindex + 1
            if size == 0:
                raise ValueError("Could not find ANY indices for slice: %s" % theslice)
            elif size == len(globalindices):
                result.append({
                    'slice' : slice(minindex, maxindex + 1, 1),
                    'edges' : theslice,
                })
            else:
                result.append({
                    'slice' : np.sort(globalindices),
                    'edges' : theslice,
                })
        return result

    def get_fluxes_shapes(self, data, axes):
        axisblocks = self.get_blocks(axes)
        if type(data) is np.ndarray:
            shapes = np.ones(data.shape, dtype=data.dtype)
            fluxes = np.empty(len(axisblocks), dtype=data.dtype)
        else:
            shapes = torch.ones(data.shape, dtype=data.dtype, device=data.device)
            fluxes = torch.ones(len(axisblocks), dtype=data.dtype, device=data.device)

        fluxbinning = Binning()
        fluxbinning.Nax = len(axes)
        fluxbinning.axis_names = axes

        for i, block in enumerate(axisblocks):
            indexing = block['slice']
            if type(data) is torch.Tensor:
                fluxes[i] *= torch.sum(data[indexing])
            else:
                fluxes[i] = np.sum(data[indexing])

            shapes[indexing] = data[indexing]/fluxes[i]

            binningblock = BinningBlock()
            binningblock.Nax = len(axes)
            binningblock.axis_names = axes
            for ax in axes:
                binningblock.extents.append(1)
                binningblock.ax_details[ax] = {
                    'edges' : block['edges'][ax],
                    'extent' : 1,
                    'minedge' : block['edges'][ax][0],
                    'maxedge' : block['edges'][ax][1],
                }
            binningblock.total_size = 1
            binningblock.offset = i
            binningblock.calculate_strides()

            fluxbinning.blocks.append(binningblock)

        return fluxes, shapes, fluxbinning

    def merge_fluxes_shapes(self, fluxes, shapes, fluxbinning):
        axes = fluxbinning.axis_names
        axisblocks = self.get_blocks(axes)

        if type(fluxes) is torch.Tensor:
            result = torch.empty(shapes.shape, dtype=shapes.dtype, device=shapes.device)
        else:
            result = np.empty(shapes.shape, dtype=shapes.dtype)

        for i, block in enumerate(axisblocks):
            indexing = block['slice']
            shape = shapes[indexing]
            flux = fluxbinning.get_slice(fluxes, **block['edges'])
            result[indexing] = shape * flux

        return result

    '''
    For internal use only. Not part of the public interface
    '''
    def get_specblock_binning(self, data, specblock):
        ranges = {name : (np.min(specblock[name]), np.max(specblock[name])) for name in self.axis_names}
        sizes = [np.max(specblock[name]) - np.min(specblock[name]) for name in self.axis_names]
        extradims = list(data.shape[1:])
        sizes = sizes + extradims
        theslice = self.blocks[0].get_slice_from_indices(data, **ranges)
        theslice = theslice.reshape(sizes)

        for i, name in enumerate(self.axis_names):
            theslice = np.add.reduceat(theslice, 
                                       np.asarray(specblock[name][:-1]) - np.min(specblock[name]),
                                       axis=i)
        return theslice

class GenRecoBinning:
    def __init__(self):
        self.genbinning = None
        self.recobinning = None

    def copy(self):
        result = GenRecoBinning()
        result.from_dict(self.to_dict())
        return result

    def setup_from_histograms(self, Hreco, Hgen):
        self.genbinning = Binning()
        self.genbinning.setup_from_histogram(Hgen)

        self.recobinning = Binning()
        self.recobinning.setup_from_histogram(Hreco)

    def dump_to_file(self, file):
        print("Writing binning spec to file: %s" % file)
        os.makedirs(os.path.dirname(file), exist_ok=True)
        resultdict = self.to_dict()
        with open(file, 'w') as f:
            json.dump(resultdict, f, indent=4)

    def load_from_file(self, file):
        print("Reading binning spec from file: %s" % file)

        with open(file, 'r') as f:
            resultdict = json.load(f)

        self.from_dict(resultdict)

    def to_dict(self):
        gendict = self.genbinning.to_dict() 
        recodict = self.recobinning.to_dict()
        resultdict = {
            'gen': gendict,
            'reco': recodict
        }
        return resultdict

    def from_dict(self, resultdict):
        self.genbinning = Binning()
        self.genbinning.from_dict(resultdict['gen'])

        self.recobinning = Binning()
        self.recobinning.from_dict(resultdict['reco'])

    def get_slice(self, data, genreco, **theedges):
        if genreco.lower().strip() == 'gen':
            thebinning = self.genbinning
        elif genreco.lower().strip() == 'reco':
            thebinning = self.recobinning
        else:
            raise ValueError("genreco must be 'gen' or 'reco'")

        return thebinning.get_slice(data, **theedges)

    def get_slice_cov2d(self, data, genreco, **theedges):
        if genreco.lower().strip() == 'gen':
            thebinning = self.genbinning
        elif genreco.lower().strip() == 'reco':
            thebinning = self.recobinning
        else:
            raise ValueError("genreco must be 'gen' or 'reco'")

        return thebinning.get_slice_cov2d(data, **theedges)

    def get_slice_transfer2d(self, data, **theedges):
        result = genbinning.get_slice(data.T, genreco='gen', **theedges)
        result = recobinning.get_slice(result.T, genreco='reco', **theedges)
        return result

    def project_out(self, data, genreco, axis_name):
        if genreco.lower().strip() == 'gen':
            thebinning = self.genbinning
        elif genreco.lower().strip() == 'reco':
            thebinning = self.recobinning
        else:
            raise ValueError("genreco must be 'gen' or 'reco'")

        return thebinning.project_out(data, axis_name)

    def project_out_cov2d(self, data, genreco, axis_name):
        if genreco.lower().strip() == 'gen':
            thebinning = self.genbinning
        elif genreco.lower().strip() == 'reco':
            thebinning = self.recobinning
        else:
            raise ValueError("genreco must be 'gen' or 'reco'")

        return thebinning.project_out_cov2d(data, axis_name)

    def project_out_transfer2d(self, data, axis_name):
        if type(data) in [list, tuple]:
            recodata = data[0]
            gendata = data[1]

            recodata, newbinning_reco = self.project_out(recodata.T, 'reco', axis_name)
            gendata, newbinning_gen = self.project_out(gendata.T, 'gen', axis_name)

            result = [recodata.T, gendata.T]
        else:
            result, newbinning_gen = self.genbinning.project_out(data.T, axis_name)
            result, newbinning_reco = self.recobinning.project_out(result.T, axis_name)

        newbinning = GenRecoBinning()
        newbinning.genbinning = newbinning_gen
        newbinning.recobinning = newbinning_reco

        return result, newbinning

    def rebin(self, data, genreco, rebinning_spec):
        if genreco.lower().strip() == 'gen':
            thebinning = self.genbinning
        elif genreco.lower().strip() == 'reco':
            thebinning = self.recobinning
        else:
            raise ValueError("genreco must be 'gen' or 'reco'")
        
        if rebinning_spec is None:
            return data, thebinning.copy()

        return thebinning.rebin(data, rebinning_spec)

    def rebin_cov2d(self, data, genreco, rebinning_spec):
        if genreco.lower().strip() == 'gen':
            thebinning = self.genbinning
        elif genreco.lower().strip() == 'reco':
            thebinning = self.recobinning
        else:
            raise ValueError("genreco must be 'gen' or 'reco'")

        if rebinning_spec is None:
            return data, thebinning

        return thebinning.rebin_cov2d(data, rebinning_spec)

    def rebin_transfer2d(self, data, rebinning_reco, rebinning_gen):
        if type(data) in [list, tuple]:
            if len(data) != 2:
                raise ValueError("Data must be a tuple or list of length 2 containing reco and gen data, or else an np.ndarray transfer matrix")

            recodata = data[0]
            gendata = data[1]

            recodata, newbinning_reco = self.rebin(recodata.T, 'reco', rebinning_reco)
            gendata, newbinning_gen = self.rebin(gendata.T, 'gen', rebinning_gen)

            result = [recodata.T, gendata.T]

        else:
            result, newbinning_gen = self.rebin(data.T, 'gen', rebinning_gen)
            result, newbinning_reco = self.rebin(result.T, 'reco', rebinning_reco) 

        newbinning = GenRecoBinning()
        newbinning.genbinning = newbinning_gen
        newbinning.recobinning = newbinning_reco

        return result, newbinning
