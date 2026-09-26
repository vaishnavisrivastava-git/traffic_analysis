#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr double kInteractionRadius = 50.0;
constexpr double kTtcThreshold = 3.0;
constexpr std::size_t kInitialReserve = 100000;
const std::vector<std::pair<std::string, std::size_t>> kWorkloadSizes = {
    {"1K", 1000}, {"5K", 5000}, {"10K", 10000}, {"50K", 50000},
    {"100K", 100000}, {"250K", 250000}, {"500K", 500000},
    {"1M", 1000000}, {"1.5M", 1500000}, {"2M", 2000000},
};

struct Record {
    long long vehicleId;
    long long frameId;
    double globalTime;
    long long laneId;
    double localX;
    double localY;
    double velocity;
    double acceleration;
};

struct Interaction {
    long long frameId;
    double globalTime;
    long long vehicleId1;
    long long vehicleId2;
    long long laneId1;
    long long laneId2;
    double distance;
    double relativeVelocity;
    double ttc;
    bool critical;
};

struct BenchmarkResult {
    std::size_t recordsProcessed = 0;
    unsigned long long candidatePairs = 0;
    unsigned long long nearbyInteractions = 0;
    unsigned long long criticalCount = 0;
    std::vector<Interaction> interactions;
    double seconds = 0.0;
};

std::vector<std::string> parseCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool insideQuotes = false;

    for (std::size_t index = 0; index < line.size(); ++index) {
        const char character = line[index];
        if (character == '"') {
            if (insideQuotes && index + 1 < line.size() && line[index + 1] == '"') {
                field.push_back('"');
                ++index;
            } else {
                insideQuotes = !insideQuotes;
            }
        } else if (character == ',' && !insideQuotes) {
            fields.push_back(field);
            field.clear();
        } else {
            field.push_back(character);
        }
    }
    fields.push_back(field);
    return fields;
}

double parseDouble(const std::string& value, const std::string& column) {
    std::size_t parsedCharacters = 0;
    const double parsed = std::stod(value, &parsedCharacters);
    if (parsedCharacters != value.size()) {
        throw std::runtime_error("Invalid numeric value in " + column + ": " + value);
    }
    return parsed;
}

long long parseInteger(const std::string& value, const std::string& column) {
    const double parsed = parseDouble(value, column);
    if (!std::isfinite(parsed) || std::trunc(parsed) != parsed) {
        throw std::runtime_error("Expected an integer in " + column + ": " + value);
    }
    return static_cast<long long>(parsed);
}

std::vector<Record> loadRecords(const std::string& inputPath, std::size_t maximumRecords) {
    std::ifstream input(inputPath);
    if (!input) {
        throw std::runtime_error("Could not open input CSV: " + inputPath);
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("Input CSV is empty: " + inputPath);
    }
    const auto header = parseCsvLine(line);
    std::unordered_map<std::string, std::size_t> columnIndexes;
    for (std::size_t index = 0; index < header.size(); ++index) {
        columnIndexes[header[index]] = index;
    }

    const std::vector<std::string> requiredColumns = {
        "Vehicle_ID", "Frame_ID", "Global_Time", "Lane_ID", "Local_X", "Local_Y",
        "v_Vel", "v_Acc"};
    for (const auto& column : requiredColumns) {
        if (columnIndexes.find(column) == columnIndexes.end()) {
            throw std::runtime_error("Input CSV is missing required column: " + column);
        }
    }

    std::vector<Record> records;
    records.reserve(std::min(maximumRecords, kInitialReserve));
    while (records.size() < maximumRecords && std::getline(input, line)) {
        const auto fields = parseCsvLine(line);
        if (fields.size() != header.size()) {
            throw std::runtime_error("CSV row has a different number of fields than its header");
        }
        records.push_back({
            parseInteger(fields[columnIndexes.at("Vehicle_ID")], "Vehicle_ID"),
            parseInteger(fields[columnIndexes.at("Frame_ID")], "Frame_ID"),
            parseDouble(fields[columnIndexes.at("Global_Time")], "Global_Time"),
            parseInteger(fields[columnIndexes.at("Lane_ID")], "Lane_ID"),
            parseDouble(fields[columnIndexes.at("Local_X")], "Local_X"),
            parseDouble(fields[columnIndexes.at("Local_Y")], "Local_Y"),
            parseDouble(fields[columnIndexes.at("v_Vel")], "v_Vel"),
            parseDouble(fields[columnIndexes.at("v_Acc")], "v_Acc"),
        });
    }
    return records;
}

double calculateDistance(const Record& first, const Record& second) {
    const double deltaX = first.localX - second.localX;
    const double deltaY = first.localY - second.localY;
    return std::sqrt(deltaX * deltaX + deltaY * deltaY);
}

BenchmarkResult analyzeRecords(const std::vector<Record>& records,
                               std::size_t recordCount,
                               bool collectInteractions = false) {
    BenchmarkResult result;
    result.recordsProcessed = recordCount;

    const auto start = std::chrono::steady_clock::now();
    std::map<long long, std::vector<const Record*>> recordsByFrame;
    for (std::size_t index = 0; index < recordCount; ++index) {
        recordsByFrame[records[index].frameId].push_back(&records[index]);
    }

    for (auto& frameEntry : recordsByFrame) {
        auto& frameRecords = frameEntry.second;
        std::stable_sort(frameRecords.begin(), frameRecords.end(),
                         [](const Record* first, const Record* second) {
                             return first->vehicleId < second->vehicleId;
                         });
        frameRecords.erase(
            std::unique(frameRecords.begin(), frameRecords.end(),
                        [](const Record* first, const Record* second) {
                            return first->vehicleId == second->vehicleId;
                        }),
            frameRecords.end());

        for (std::size_t firstIndex = 0; firstIndex < frameRecords.size(); ++firstIndex) {
            for (std::size_t secondIndex = firstIndex + 1; secondIndex < frameRecords.size(); ++secondIndex) {
                const Record& first = *frameRecords[firstIndex];
                const Record& second = *frameRecords[secondIndex];
                ++result.candidatePairs;

                if (std::llabs(first.laneId - second.laneId) > 1) {
                    continue;
                }

                const double separation = calculateDistance(first, second);
                if (separation > kInteractionRadius) {
                    continue;
                }

                const double relativeSpeed = first.velocity - second.velocity;
                const double ttc = relativeSpeed < 0.0
                                       ? separation / std::abs(relativeSpeed)
                                       : std::numeric_limits<double>::infinity();
                const bool critical = std::isfinite(ttc) && ttc <= kTtcThreshold;
                ++result.nearbyInteractions;
                if (critical) {
                    ++result.criticalCount;
                }
                if (collectInteractions) {
                    result.interactions.push_back({
                        frameEntry.first,
                        first.globalTime,
                        first.vehicleId,
                        second.vehicleId,
                        first.laneId,
                        second.laneId,
                        separation,
                        relativeSpeed,
                        ttc,
                        critical,
                    });
                }
            }
        }
    }

    const auto end = std::chrono::steady_clock::now();
    result.seconds = std::chrono::duration<double>(end - start).count();
    if (collectInteractions) {
        std::sort(result.interactions.begin(), result.interactions.end(),
                  [](const Interaction& first, const Interaction& second) {
                      if (first.frameId != second.frameId) {
                          return first.frameId < second.frameId;
                      }
                      if (first.vehicleId1 != second.vehicleId1) {
                          return first.vehicleId1 < second.vehicleId1;
                      }
                      return first.vehicleId2 < second.vehicleId2;
                  });
    }
    return result;
}

void writeInteractions(const std::string& outputPath, const std::vector<Interaction>& interactions) {
    std::ofstream output(outputPath);
    if (!output) {
        throw std::runtime_error("Could not write interaction CSV: " + outputPath);
    }
    output << "Frame_ID,Global_Time,Vehicle_ID_1,Vehicle_ID_2,Lane_ID_1,Lane_ID_2,"
              "Distance,Relative_Velocity,TTC,Critical_Interaction\n";
    output << std::setprecision(17);
    for (const auto& item : interactions) {
        output << item.frameId << ',' << item.globalTime << ',' << item.vehicleId1 << ','
               << item.vehicleId2 << ',' << item.laneId1 << ',' << item.laneId2 << ','
               << item.distance << ',' << item.relativeVelocity << ',' << item.ttc << ','
               << (item.critical ? "True" : "False") << '\n';
    }
}

void writeResults(const std::string& outputPath,
                  std::size_t availableRecords,
                  const std::vector<std::pair<std::string, BenchmarkResult>>& results) {
    const std::filesystem::path outputFile(outputPath);
    if (!outputFile.parent_path().empty()) {
        std::filesystem::create_directories(outputFile.parent_path());
    }
    std::ofstream output(outputPath);
    if (!output) {
        throw std::runtime_error("Could not write benchmark results: " + outputPath);
    }

    output << "Sequential CPU Baseline (C++17)\n"
              "===============================\n"
                  "Input: data/processed/ngsim_cleaned.csv\n"
                  "Execution time covers frame grouping and pair analysis; CSV loading, output, and console I/O are excluded.\n"
                  "Each workload uses a deterministic prefix of the cleaned CSV.\n"
              << "Records available: " << availableRecords << "\n\n";
    output << std::fixed << std::setprecision(6);
    for (std::size_t index = 0; index < results.size(); ++index) {
          const auto& result = results[index].second;
          output << "Workload: " << results[index].first << '\n'
               << "Records processed: " << result.recordsProcessed << '\n'
               << "Candidate pairs: " << result.candidatePairs << '\n'
                    << "Nearby interactions: " << result.nearbyInteractions << '\n'
                    << "Critical interactions: " << result.criticalCount << '\n'
               << "Execution time (seconds): " << result.seconds << "\n\n";
    }
}

struct Arguments {
    std::string inputPath = "data/processed/ngsim_cleaned.csv";
    std::string resultsPath = "results/sequential_results.txt";
    std::string interactionsPath;
    std::size_t verifyLimit = 0;
};

Arguments parseArguments(int argumentCount, char* argumentValues[]) {
    Arguments arguments;
    for (int index = 1; index < argumentCount; ++index) {
        const std::string option = argumentValues[index];
        if (index + 1 >= argumentCount) {
            throw std::runtime_error("Missing value for command-line option: " + option);
        }
        const std::string value = argumentValues[++index];
        if (option == "--input") {
            arguments.inputPath = value;
        } else if (option == "--results") {
            arguments.resultsPath = value;
        } else if (option == "--interactions") {
            arguments.interactionsPath = value;
        } else if (option == "--verify-limit") {
            arguments.verifyLimit = static_cast<std::size_t>(std::stoull(value));
        } else {
            throw std::runtime_error("Unknown command-line option: " + option);
        }
    }
    return arguments;
}

}  // namespace

int main(int argumentCount, char* argumentValues[]) {
    try {
        const Arguments arguments = parseArguments(argumentCount, argumentValues);
        if (arguments.verifyLimit > 0) {
            if (arguments.interactionsPath.empty()) {
                throw std::runtime_error("--verify-limit requires --interactions <path>");
            }
            const auto records = loadRecords(arguments.inputPath, arguments.verifyLimit);
            const auto result = analyzeRecords(records, records.size(), true);
            writeInteractions(arguments.interactionsPath, result.interactions);
            std::cout << "Verification subset records: " << result.recordsProcessed << '\n'
                      << "Candidate pairs: " << result.candidatePairs << '\n'
                      << "Nearby interactions: " << result.nearbyInteractions << '\n'
                      << "Critical interactions: " << result.criticalCount << '\n';
            return 0;
        }

        const auto records = loadRecords(arguments.inputPath, std::numeric_limits<std::size_t>::max());
        std::vector<std::pair<std::string, BenchmarkResult>> labeledResults;
        labeledResults.reserve(kWorkloadSizes.size() + 1);

        for (const auto& workload : kWorkloadSizes) {
            const std::size_t requestedSize = workload.second;
            const std::size_t actualSize = std::min(requestedSize, records.size());
            const auto result = analyzeRecords(records, actualSize);
            labeledResults.emplace_back(workload.first, result);
        }
        const auto fullResult = analyzeRecords(records, records.size());
        labeledResults.emplace_back("Full cleaned dataset", fullResult);

        writeResults(arguments.resultsPath, records.size(), labeledResults);

        std::cout << "Benchmark results written to: " << arguments.resultsPath << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}