/*
 * Cornerstone octree
 *
 * Copyright (c) 2026 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Point-to-point exchange of global tree quantities, replaces the Allgatherv of global leaves
 *
 * @author Yannik Rüfenacht <yannik.rue@gmail.com>
 *
 * Overall procedure:
 *      1.  Upsweep: per parent level, bottom up, children are sent to the owner of their parent,
 *          the owner combines them. Afterwards, each rank has valid quantities for the global nodes
 *          it owns and their children. Send/recv lists follow from the global tree and node owners,
 *          no indices are exchanged.
 *
 *      2.  Extract: each rank requests the global nodes of its far LET leaves from their owners
 *          and receives their quantities.
 */

#pragma once

#include <algorithm>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/tree/octree.hpp"

namespace cstone
{
/*! @brief owner rank of every global tree node
 *
 * @param[in]  gOctree    global octree, same on all ranks
 * @param[in]  globDispl  first global leaf per rank, length numRanks + 1
 * @param[out] nodeOwner  owner rank per node, length gOctree.numNodes
 */
template<class KeyType>
void computeNodeOwners(OctreeView<const KeyType> gOctree,
                       std::span<const TreeNodeIndex> globDispl,
                       std::span<int> nodeOwner)
{
    int numRanks = int(globDispl.size()) - 1;

    auto leafToInternal = gOctree.leafToInternalSpan();
    //! leaves: compute owner for each global leaf
    for (int r = 0; r < numRanks; ++r)
    {
        for (TreeNodeIndex i = globDispl[r]; i < globDispl[r + 1]; ++i)
        {
            TreeNodeIndex nodeIdx = leafToInternal[i];
            nodeOwner[nodeIdx]    = r;
        }
    }

    //! internal nodes: compute owner for each internal node
    for (TreeNodeIndex i = gOctree.numNodes - 1; i >= 0; --i)
    {
        TreeNodeIndex c = gOctree.childOffsets[i];
        if (c == 0) { continue; }
        nodeOwner[i] = nodeOwner[c];
    }
}

/*! @brief send/recv lists of the upsweep per parent level
 *
 * send[l]: children I own, parent owned by peer
 * recv[l]: children owned by peer, parent owned by me
 * Each message refers to the slice [offset, offset + count) of sendIdx or recvIdx, which hold the child node
 * indices of all levels in a row.
 */
struct UpsweepLists
{
    struct Message
    {
        int           peer;
        TreeNodeIndex offset;
        TreeNodeIndex count;
    };
    using Messages = std::vector<Message>;

    std::vector<Messages>      send, recv;
    std::vector<TreeNodeIndex> sendIdx, recvIdx;
};

/*! @brief send/recv lists of the upsweep
 *
 * @param[in]  gOctree    global octree, same on all ranks
 * @param[in]  nodeOwner  owner rank per node
 * @param[in]  myRank     executing rank
 * @return                send/recv lists per parent level
 */
template<class KeyType>
UpsweepLists computeUpsweepLists(OctreeView<const KeyType> gOctree, std::span<const int> nodeOwner, int myRank)
{
    auto levelRange     = gOctree.levelRangeSpan();
    int numParentLevels = maxDepth(levelRange.data(), levelRange.size()); // deepest level has only leaves

    UpsweepLists lists;
    lists.send.resize(numParentLevels);
    lists.recv.resize(numParentLevels);

    //! owners only increase along a level, so a peer's children always come in a row
    auto addMessage = [](UpsweepLists::Messages& msgs, std::vector<TreeNodeIndex>& idx, int peer, TreeNodeIndex first,
                         TreeNodeIndex last)
    {
        if (msgs.empty() || msgs.back().peer != peer) { msgs.push_back({peer, TreeNodeIndex(idx.size()), 0}); }
        for (TreeNodeIndex i = first; i < last; ++i)
        {
            idx.push_back(i);
        }
        msgs.back().count += last - first;
    };

    for (int l = 0; l < numParentLevels; ++l)
    {
        TreeNodeIndex firstParent = levelRange[l];
        TreeNodeIndex lastParent  = levelRange[l + 1];

        for (TreeNodeIndex p = firstParent; p < lastParent; ++p)
        {
            TreeNodeIndex firstChild = gOctree.childOffsets[p];
            TreeNodeIndex lastChild  = firstChild + 8;
            if (firstChild == 0) { continue; }

            //! children of p grouped by owner
            for (TreeNodeIndex c = firstChild; c < lastChild;)
            {
                int owner            = nodeOwner[c];
                TreeNodeIndex runEnd = c;
                while (runEnd < lastChild && nodeOwner[runEnd] == owner)
                {
                    ++runEnd;
                }

                if (owner == myRank && nodeOwner[p] != myRank)
                {
                    addMessage(lists.send[l], lists.sendIdx, nodeOwner[p], c, runEnd);
                }
                if (owner != myRank && nodeOwner[p] == myRank)
                {
                    addMessage(lists.recv[l], lists.recvIdx, owner, c, runEnd);
                }

                c = runEnd;
            }
        }
    }

    return lists;
}

/*! @brief p2p upsweep of the global tree
 *
 * @param[in]  gOctree          global octree, same on all ranks
 * @param[in]  globDispl        first global leaf per rank, length numRanks + 1
 * @param[in]  lists            send/recv lists per parent level
 * @param[in]  gLeafQLoc        quantities of my global leaves
 * @param[out] globQ            quantities of the global tree, valid for nodes I own and their children
 * @param[in]  myRank           executing rank
 * @param[in]  comm             MPI communicator
 * @param[in]  upsweepFunction  upsweep of one level
 * @param[in]  upsweepArgs      extra upsweep arguments, e.g. expansion centers
 */
template<class Q, class KeyType, class F, class... UArgs>
void globalUpsweep(OctreeView<const KeyType> gOctree,
                   std::span<const TreeNodeIndex> globDispl,
                   const UpsweepLists& lists,
                   std::span<const Q> gLeafQLoc,
                   std::span<Q> globQ,
                   int myRank,
                   MPI_Comm comm,
                   F&& upsweepFunction,
                   UArgs&&... upsweepArgs)
{
    auto levelRange     = gOctree.levelRangeSpan();
    auto leafToInternal = gOctree.leafToInternalSpan();
    int numParentLevels = int(lists.send.size());
    std::span<const TreeNodeIndex> sendIdx(lists.sendIdx), recvIdx(lists.recvIdx);

    //! scatter local leaf quantities
    scatter(leafToInternal.subspan(globDispl[myRank], gLeafQLoc.size()), gLeafQLoc.data(), globQ.data());

    for (int l = numParentLevels - 1; l >= 0; --l)
    {
        const auto& send = lists.send[l];
        const auto& recv = lists.recv[l];
        std::vector<std::vector<Q>> sendBuf(send.size()), recvBuf(recv.size());
        std::vector<MPI_Request> requests;
        int tag = static_cast<int>(P2pTags::focusGlobalUpsweep) + l;

        //! exchange child quantities with their owners
        for (std::size_t m = 0; m < recv.size(); ++m)
        {
            recvBuf[m].resize(recv[m].count);
            mpiRecvAsync(recvBuf[m].data(), recv[m].count, recv[m].peer, tag, requests, comm);
        }

        for (std::size_t m = 0; m < send.size(); ++m)
        {
            sendBuf[m].resize(send[m].count);
            gather(sendIdx.subspan(send[m].offset, send[m].count), globQ.data(), sendBuf[m].data());
            mpiSendAsync(sendBuf[m].data(), send[m].count, send[m].peer, tag, requests, comm);
        }

        MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

        //! place received child quantities
        for (std::size_t m = 0; m < recv.size(); ++m)
        {
            scatter(recvIdx.subspan(recv[m].offset, recv[m].count), recvBuf[m].data(), globQ.data());
        }

        // TODO GPU does not work with subspans
        //! accumulate this parent level
        upsweepFunction(levelRange.subspan(l, 2), gOctree.childOffsets, globQ.data(), upsweepArgs...);
    }
}

/*! @brief gather with distributed src, src[gmap[k]] fetched from its owner rank
 *
 * @param[in]  gmap       global node per entry
 * @param[in]  nodeOwner  owner rank per global node
 * @param[in]  src        quantities of the global tree, valid on the owner
 * @param[out] out        out[k] = src[gmap[k]], length gmap.size()
 * @param[in]  comm       MPI communicator
 */
template<class T>
void gatherFromOwners(
    std::span<const TreeNodeIndex> gmap, std::span<const int> nodeOwner, const T* src, T* out, MPI_Comm comm)
{
    int numRanks = 0;
    MPI_Comm_size(comm, &numRanks);
    std::size_t n = gmap.size();

    int reqTag   = static_cast<int>(P2pTags::focusGlobalExtract);
    int valueTag = reqTag + 1;

    //! round 1: global nodes I need, grouped by owner, counts of round 1 (become recv counts in round 2)
    std::vector<int> sendCounts(numRanks, 0), sendDispl(numRanks + 1, 0);
    std::vector<TreeNodeIndex> sendIdx(n);

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b)
              { return std::pair(nodeOwner[gmap[a]], gmap[a]) < std::pair(nodeOwner[gmap[b]], gmap[b]); });

    //! send buffer and counts per owner
    for (std::size_t m = 0; m < n; ++m)
    {
        TreeNodeIndex g = gmap[order[m]];
        sendIdx[m]      = g;
        ++sendCounts[nodeOwner[g]];
    }
    std::inclusive_scan(sendCounts.begin(), sendCounts.end(), sendDispl.begin() + 1);

    //! number of requests each rank sends to me
    std::vector<int> recvCounts(numRanks, 0), recvDispl(numRanks + 1, 0);
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, comm);
    std::inclusive_scan(recvCounts.begin(), recvCounts.end(), recvDispl.begin() + 1);

    //! exchange global node indices, sendIdx to owners, requesters to recvIdx
    std::vector<TreeNodeIndex> recvIdx(recvDispl[numRanks]);
    std::vector<MPI_Request> requests;

    for (int r = 0; r < numRanks; ++r)
    {
        if (recvCounts[r] > 0)
        {
            mpiRecvAsync(recvIdx.data() + recvDispl[r], recvCounts[r], r, reqTag, requests, comm);
        }
    }

    for (int r = 0; r < numRanks; ++r)
    {
        if (sendCounts[r] > 0)
        {
            mpiSendAsync(sendIdx.data() + sendDispl[r], sendCounts[r], r, reqTag, requests, comm);
        }
    }

    MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    //! round 2: values of the requested nodes
    std::vector<T> sendQ(recvIdx.size()), recvQ(n);
    for (std::size_t i = 0; i < recvIdx.size(); ++i)
    {
        sendQ[i] = src[recvIdx[i]];
    }

    requests.clear();
    for (int r = 0; r < numRanks; ++r)
    {
        if (sendCounts[r] > 0)
        {
            mpiRecvAsync(recvQ.data() + sendDispl[r], sendCounts[r], r, valueTag, requests, comm);
        }
    }

    for (int r = 0; r < numRanks; ++r)
    {
        if (recvCounts[r] > 0)
        {
            mpiSendAsync(sendQ.data() + recvDispl[r], recvCounts[r], r, valueTag, requests, comm);
        }
    }

    MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    //! recvQ is in sendIdx order, order[m] is the original position
    for (std::size_t m = 0; m < n; ++m)
    {
        out[order[m]] = recvQ[m];
    }
}

} // namespace cstone
