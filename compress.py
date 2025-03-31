"""
Assignment 2 starter code
CSC148, Winter 2025
Instructors: Bogdan Simion, Rutwa Engineer, Marc De Benedetti, Romina Piunno

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2025 Bogdan Simion, Dan Zingaro
"""
from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq_dict = {}
    for by in text:
        if by not in freq_dict:
            freq_dict[by] = 1
        else:
            freq_dict[by] += 1

    return freq_dict


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    trees = []
    c = 0

    for sym, freq in freq_dict.items():
        trees.append((freq, c, HuffmanTree(sym)))
        c += 1

    if len(trees) == 1:
        sym = next(iter(freq_dict))
        temp_sym = (sym + 1) % 256
        temp_tree = HuffmanTree(temp_sym)
        return HuffmanTree(None, trees[0][2], temp_tree)

    while len(trees) > 1:
        trees.sort(key=lambda  x: (x[0], x[1]))
        freq1, _, tree1 = trees.pop(0)
        freq2, _, tree2 = trees.pop(0)
        combined_tree = HuffmanTree(None, tree1, tree2)
        trees.append((freq1 + freq2, c, combined_tree))
        c += 1

    return trees[0][2]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    codes = {}
    _get_codes_helper(tree, "", codes)
    return codes


def _get_codes_helper(node: HuffmanTree, path: str, codes: dict[int, str]) -> None:
    """Recursive helper that fills in <codes> with paths to leaves."""
    if node.is_leaf():
        codes[node.symbol] = path
    else:
        if node.left:
            _get_codes_helper(node.left, path + '0', codes)
        if node.right:
            _get_codes_helper(node.right, path + '1', codes)


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    _number_nodes_helper(tree, [0])

def _number_nodes_helper(node: HuffmanTree, counter: list[int]) -> None:
    if node is None or node.is_leaf():
        return

    if node.left:
        _number_nodes_helper(node.left, counter)
    if node.right:
        _number_nodes_helper(node.right, counter)

    node.number = counter[0]
    counter[0] += 1


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    code = get_codes(tree)
    total_freq = 0
    weighted_len = 0

    for sym, freq, in freq_dict.items():
        code_len = len(code[sym])
        weighted_len += freq * code_len
        total_freq += freq

    return weighted_len / total_freq


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    bit_string = ''
    for by in text:
        bit_string += codes[by]

    if len(bit_string) % 8 != 0:
        padding = 8 - (len(bit_string) % 8)
        bit_string += "0" * padding

    ret = []
    for i in range(0, len(bit_string), 8):
        byte = bit_string[i:i+8]
        ret.append(bits_to_byte(byte))

    return bytes(ret)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    internal_nodes = []
    _postorder_internal(tree, internal_nodes)

    ret = []
    for node in internal_nodes:
        if node.left is not None:
            if node.left.is_leaf():
                left_flag = 0
                left_val = node.left.symbol
            else:
                left_flag = 1
                left_val = node.left.number
        else:
            left_flag, left_val = 0, 0

        if node.right is not None:
            if node.right.is_leaf():
                right_flag = 0
                right_val = node.right.symbol
            else:
                right_flag = 1
                right_val = node.right.number
        else:
            right_flag, right_val = 0, 0

        ret.extend([left_flag, left_val, right_flag, right_val])

    return bytes(ret)


def _postorder_internal(node: HuffmanTree, nodes: list[HuffmanTree]) -> None:
    """Helper function that appends all internal nodes of <node> to <nodes> in postorder.

    Internal nodes are those for which node.is_leaf() returns False.
    """
    if node is None:
        return

    if node.left:
        _postorder_internal(node.left, nodes)
    if node.right:
        _postorder_internal(node.right, nodes)

    if not node.is_leaf():
        nodes.append(node)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    node = node_lst[root_index]

    if node.l_type == 0:
        left_tree = HuffmanTree(node.l_data)
    else:
        left_tree = generate_tree_general(node_lst, node.l_data)

    if node.r_type == 0:
        right_tree = HuffmanTree(node.r_data)
    else:
        right_tree = generate_tree_general(node_lst, node.r_data)

    return HuffmanTree(None, left_tree, right_tree)


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    ret = []
    for i in range(root_index + 1):
        node = node_lst[i]
        if node.r_type == 1:
            right_child = ret.pop()
        else:
            right_child = HuffmanTree(node.r_data)

        if node.l_type == 1:
            light_child = ret.pop()
        else:
            light_child = HuffmanTree(node.l_data)

        new_tree = HuffmanTree(None, light_child, right_child)
        ret.append(new_tree)

    return ret[-1]

def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    bit_str = "".join(byte_to_bits(byte) for byte in text)

    result = []
    curr = tree

    for bit in bit_str:
        if bit == '0':
            curr = curr.left
        else:
            curr = curr.right

        if curr.is_leaf():
            result.append(curr.symbol)
            if len(result) == size:
                return bytes(result)
            curr = tree

    return  bytes(result)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    leaves = []
    _collect_leaves(tree, 0, leaves)

    #Shallow first
    leaves.sort(key=lambda pair: pair[1])

    flag = True
    while flag:
        flag = False
        n = len(leaves)
        for i in range(n):
            for j in range(i + 1, n):
                if leaves[i][1] < leaves[j][1]:
                    leaf_i, _ = leaves[i]
                    leaf_j, _ = leaves[j]
                    if freq_dict[leaf_i.symbol] < freq_dict[leaf_j.symbol]:
                        leaf_i.symbol, leaf_j.symbol = leaf_j.symbol, leaf_i.symbol
                        changed = True

def _collect_leaves(node: HuffmanTree, depth: int, leaves: list[tuple[HuffmanTree, int]]) -> None:
    """Recursively collect all leaves in <node> along with their depth."""
    if node is None:
        return
    if node.is_leaf():
        leaves.append((node, depth))
    else:
        _collect_leaves(node.left, depth + 1, leaves)
        _collect_leaves(node.right, depth + 1, leaves)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
