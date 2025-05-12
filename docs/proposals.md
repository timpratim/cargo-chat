# Proposals

## Index Freshness Tracking

**Problem:** Currently, the index (`index.bin`) is generated manually via the `index` command. There is no built-in mechanism to determine if the index is stale relative to the source code repository it represents.

**Proposal:** Enhance the indexing process and the stored index data to include metadata about the index's state:

1.  **Timestamp:** Record the UTC timestamp when the index generation process completes successfully.
2.  **Source Commit Hash:** For repositories managed by Git, record the full commit hash (`git rev-parse HEAD`) of the repository state that was indexed.

**Implementation:**

*   **Metadata Storage:** Add `creation_timestamp: String` (ISO 8601 format) and `source_commit_hash: Option<String>` fields to the `Ann` struct in `src/ann.rs`. This ensures the metadata is stored directly within the `index.bin` file alongside the vectors and chunk metadata.
*   **Serialization:** Update the `serde::Serialize` and `serde::Deserialize` implementations for the `Ann` struct to handle the new fields.
*   **Indexing Logic (`src/main.rs`):**
    *   Modify the `execute_index_command` function to:
        *   Retrieve the current UTC timestamp before serialization.
        *   Attempt to retrieve the current Git commit hash of the target repository (`repo_path`) using `std::process::Command` to run `git rev-parse HEAD`. Store it as an `Option<String>`, handling cases where the command fails or the directory isn't a Git repository.
        *   Pass the timestamp and commit hash to the `Ann::build` function (which will also need modification).
*   **Display:** Update the `ReplSubCmd::Status` handler in `src/main.rs` to display the timestamp and commit hash if an index is loaded.

**Benefits:**

*   Provides clear information about *when* the index was created.
*   Provides clear information about *what* state of the code the index represents (via commit hash).
*   Enables future functionality to automatically check for staleness by comparing the stored commit hash with the current repository hash, or by comparing file modification times against the stored timestamp.
*   Stores metadata atomically with the index data.

**Future Considerations:**

*   Implement an automatic check (e.g., on query or via a dedicated command) that compares the stored metadata against the current repository state to determine if reindexing is needed.
*   Explore file checksums as an alternative or supplement for non-Git repositories. 

## Automatic Index Staleness Check

**Goal:** Provide a mechanism to automatically detect if the loaded index is potentially stale compared to the source repository state and inform the user.

**Proposed Implementation:**

1.  **Add a `check-index` command:**
    *   Introduce a new subcommand to the REPL (e.g., `check-index`) and potentially a standalone command-line argument.
    *   This command will perform the staleness check against the currently loaded index and the associated repository path.

2.  **Staleness Check Logic (within `check-index` handler or a shared function):**
    *   **Pre-requisite:** An index must be loaded (`session_state.ann_index` is `Some`). The original repository path used for indexing is needed. Currently, this path isn't stored directly in the index metadata. We might need to:
        *   a) Store the `repo_path` alongside the index metadata (similar to timestamp/commit hash).
        *   b) Infer it from `session_state.current_index_path` if the convention is that the index name relates directly to the repo name (less robust).
        *   c) Require the user to specify the repository path when running `check-index`.
        *   **(Recommendation: Store `repo_path` in the index metadata for explicitness)**
    *   **Retrieve Metadata:** Access the `creation_timestamp` and `source_commit_hash` from the loaded `ann_index`.
    *   **Get Current Repo State:**
        *   Attempt to get the current Git commit hash of the repository path using the existing `get_git_commit_hash` function.
    *   **Comparison:**
        *   **If `source_commit_hash` exists in metadata AND current commit hash was retrieved:** Compare the hashes. If they differ, the index is stale.
        *   **If `source_commit_hash` does NOT exist (e.g., not a Git repo) OR current commit hash could not be retrieved:**
            *   Fall back to timestamp comparison. Iterate through the files in the repository path (using `walkdir`, respecting `.gitignore`).
            *   Get the last modified timestamp for each relevant file.
            *   If any file's modification timestamp is later than the index's `creation_timestamp`, the index is potentially stale.
        *   **If neither commit hash nor timestamp comparison is possible:** Report that staleness cannot be determined.
    *   **Output:** Print a clear message indicating whether the index is up-to-date, stale (based on commit hash or timestamp), or if the check could not be performed reliably.

3.  **(Optional) Automatic Check on Query:**
    *   Modify the `execute_query_command` function.
    *   Before performing the similarity search, run the staleness check logic described above.
    *   If the index is found to be stale, print a warning message to the user (e.g., `WARN: Index may be stale. Consider re-indexing using the 'index' command.`) before proceeding with the query.
    *   This requires careful handling of the repository path, potentially storing it in `SessionState` when an index is loaded.

**Considerations:**

*   **Performance:** Timestamp comparison requires walking the file tree, which can be slow for very large repositories. The check on query should be efficient.
*   **Accuracy:** Timestamp comparison is less accurate than commit hash comparison (e.g., changing branches without modifying files).
*   **User Experience:** Clearly communicate the reason for potential staleness (commit mismatch vs. newer files). Provide clear instructions on how to re-index. 

## File System Watching for Automatic Reindexing

**Goal:** Automatically detect changes within the indexed source repository in near real-time and trigger a reindexing process or notify the user that the index is stale.

**Distinction from Automatic Check:** This approach uses a *persistent background process* to monitor the file system, unlike the on-demand check which compares state only when explicitly invoked or before a query.

**Proposed Implementation:**

1.  **Watcher Process:**
    *   Integrate a file system watching library (e.g., `notify` crate).
    *   When an index is loaded or the application starts with a target repository configured, potentially spawn a dedicated thread or async task to monitor the repository path associated with the index.
    *   The watcher needs to be configured to monitor for relevant events (file creation, deletion, modification, renaming).

2.  **Event Handling:**
    *   **Filtering:** Ignore events irrelevant to indexing (e.g., changes within `.git` directory, temporary editor files, build artifacts based on `.gitignore` or specific rules).
    *   **Debouncing:** File saves often trigger multiple events in quick succession. Implement debouncing logic (e.g., wait for a short period of inactivity after an event before processing) to avoid redundant triggers.
    *   **Thresholding (Optional):** Define a threshold (e.g., number of changed files, type of changes) to decide if a detected change warrants a full reindex versus perhaps a notification.

3.  **Triggering Logic:**
    *   Upon detecting significant, debounced changes:
        *   **Option A (Notification):** Update the application state (e.g., a flag in `SessionState`) to indicate the index is stale. The REPL prompt or status command could reflect this. A log message could also be generated.
        *   **Option B (Automatic Reindexing):** Trigger the `execute_index_command` function automatically. This requires careful consideration of resource usage and potential disruption to ongoing user activities (like querying).
        *   **(Recommendation: Start with Notification (Option A) due to complexity and resource implications of automatic reindexing).**

4.  **Configuration:**
    *   Provide configuration options (e.g., enable/disable watching, debounce duration, ignored paths/patterns) potentially via command-line flags or a configuration file.

**Benefits:**

*   Potentially near real-time detection of repository changes.
*   Can enable fully automated reindexing (if Option B is chosen).
*   Proactive notification to the user about index staleness.

**Cons:**

*   **Resource Usage:** Continuous file system monitoring consumes system resources (CPU, file handles).
*   **Complexity:** Implementing robust event handling (filtering, debouncing, error handling) is significantly more complex than on-demand checks.
*   **Platform Differences:** File system event behavior can vary across operating systems.
*   **Potential for Noise:** Might trigger too often on frequent saves or minor changes if not carefully tuned.
*   Managing the lifecycle of the watcher process/thread adds complexity. 