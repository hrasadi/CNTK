//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include <random>

#include "LocalTimelineBlockRandomizer.h"
#include "RandomOrdering.h"
#include <tuple>

namespace CNTK {

LocalTimelineBlockRandomizer::LocalTimelineBlockRandomizer(
    DataDeserializerPtr deserializer,
    bool sampleBasedRandomizationWindow,
    size_t randomizationRange,
    size_t seedOffset,
    bool multithreadedGetNextSequences,
    size_t maxNumberOfInvalidSequences)
: Base(deserializer, multithreadedGetNextSequences, maxNumberOfInvalidSequences),
  m_randomizationRange(randomizationRange),
  m_seedOffset(seedOffset),
  m_chunkPosition(0),
  m_sampleBasedRandomizationWindow(sampleBasedRandomizationWindow)
{
    m_prefetchedChunkDescriptions = m_originalChunkDescriptions;
    m_rng.seed((unsigned long)m_sweepIndex + m_seedOffset);
    Microsoft::MSR::CNTK::RandomShuffleMT(m_prefetchedChunkDescriptions, m_rng);
}

void LocalTimelineBlockRandomizer::PrefetchChunks()
{
    size_t capturedPosition = m_chunkPosition;
    size_t capturedSweepIndex = m_sweepIndex;

    m_prefetch = std::async(std::launch::async, [=]()
    {
        size_t position = capturedPosition;
        size_t sweepIndex = capturedSweepIndex;
        // Prefetch does not change any state that cannot be recalculated,
        // only prefetches data.
        int64_t range = m_randomizationRange;
        m_prefetchedChunks.clear();
        m_prefetchedSequences.clear();
        while (range > 0)
        {
            if (position == 0)
            {
                sweepIndex++;
                m_prefetchedChunkDescriptions = m_originalChunkDescriptions;
                m_rng.seed((unsigned long)sweepIndex + m_seedOffset);
                Microsoft::MSR::CNTK::RandomShuffleMT(m_prefetchedChunkDescriptions, m_rng);
            }

            auto desc = m_prefetchedChunkDescriptions[position];
            if (position % m_config.m_numberOfWorkers == m_config.m_workerRank) // Need to add to the window
            {
                ChunkPtr data;
                size_t oldSize = m_prefetchedSequences.size();
                if (m_window.m_dataChunks.find(desc.m_id) == m_window.m_dataChunks.end())
                {
                    // Query deserializer.
                    data = m_deserializer->GetChunk(desc.m_id);
                    m_deserializer->GetSequencesForChunk(desc.m_id, m_prefetchedSequences);
                }
                else // Simple copy
                {
                    for (size_t i = 0; i < m_window.m_sequences.size(); ++i)
                        if (m_window.m_sequences[i].m_chunkId == desc.m_id)
                            m_prefetchedSequences.push_back(m_window.m_sequences[i]);
                    data = m_window.m_dataChunks[desc.m_id];
                }

                m_prefetchedChunks.push_back(std::make_tuple(desc, data));

                if (m_sampleBasedRandomizationWindow)
                    --range;
                else
                    for (size_t i = oldSize; i < m_prefetchedSequences.size(); ++i)
                        range -= m_prefetchedSequences[i].m_numberOfSamples;
            }
            else
            {
                // Empty, we do not need data , only for tracking the current chunk.
                m_prefetchedChunks.push_back(std::make_tuple(ChunkDescription{}, nullptr));
            }

            position = (position + 1) % m_originalChunkDescriptions.size();
        }

        // Find all end of sweep markers and randomize among them.
        if (sweepIndex == capturedSweepIndex)
        {
            m_rng.seed((unsigned long)(capturedPosition + sweepIndex + m_seedOffset));
            Microsoft::MSR::CNTK::RandomShuffleMT(m_prefetchedSequences, m_rng);
        }
        else
        {
            std::vector<std::pair<size_t, size_t>> sweepIndices;
            size_t curPos = 0;
            for (size_t i = 0; i < m_prefetchedSequences.size(); ++i)
                if (IsEndOfSweep(m_prefetchedSequences[i]))
                {
                    sweepIndices.push_back(std::make_pair(curPos, i));
                    curPos = i + 1;
                }

            sweepIndices.push_back(std::make_pair(curPos, m_prefetchedSequences.size()));

            for (size_t i = 0; i < sweepIndices.size(); ++i)
            {
                m_rng.seed((unsigned long)(capturedPosition + sweepIndex + i + m_seedOffset));
                Microsoft::MSR::CNTK::RandomShuffleMT(m_prefetchedSequences, sweepIndices[i].first, sweepIndices[i].second, m_rng);
            }
        }
    });
}

void LocalTimelineBlockRandomizer::RefillSequenceWindow()
{
    if (!m_prefetch.valid())
        PrefetchChunks();

    m_prefetch.wait();

    m_window.m_sequences.clear();
    m_window.m_dataChunks.clear();

    for (const auto& c : m_prefetchedChunks)
    {
        m_window.m_sequences.insert(m_window.m_sequences.end(), m_prefetchedSequences.begin(), m_prefetchedSequences.end());
        m_window.m_dataChunks.insert(std::make_pair(std::get<0>(c).m_id, std::get<1>(c)));

        // Last chunk
        auto sweepEnd = (m_chunkPosition == m_originalChunkDescriptions.size() - 1);
        if (sweepEnd)
            m_window.m_sequences.push_back(s_endOfSweep);

        m_chunkPosition = (m_chunkPosition + 1) % m_originalChunkDescriptions.size();
    }

    // Prefetch new data chunks.
    PrefetchChunks();
}

Dictionary LocalTimelineBlockRandomizer::GetInnerState()
{
    Dictionary state;
    state[L"chunkPosition"] = (size_t)m_chunkPosition;
    return state;
}

void LocalTimelineBlockRandomizer::SetInnerState(const Dictionary& state)
{
    if (m_prefetch.valid())
        m_prefetch.wait();

    m_rng.seed((unsigned long)m_sweepIndex + m_seedOffset);
    Microsoft::MSR::CNTK::RandomShuffleMT(m_prefetchedChunkDescriptions, m_rng);
    m_chunkPosition = (ChunkIdType)state[L"globalChunkPosition"].Value<size_t>();
}

}
