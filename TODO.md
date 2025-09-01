# TODO - Semantic Network System

## Immediate Next Steps

### 🧪 Testing Improvements
- [ ] Fix the one failing test (`test_cmd_pipeline_keep_temp`) - mock Path division operation properly
- [ ] Add integration tests that use real temp files instead of mocking everything
- [ ] Test the actual CLI end-to-end with sample data
- [ ] Add performance tests for large datasets

### 📚 Documentation Updates
- [ ] Update README.md examples to use the new API
- [ ] Update CLAUDE.md with new architecture patterns
- [ ] Add API documentation with usage examples
- [ ] Create migration guide from old CLI usage to new

### 🔧 API Enhancements
- [ ] Add validation for embedding dimensions consistency
- [ ] Implement async/await support for LLM providers
- [ ] Add progress callbacks to the fluent API
- [ ] Support for custom similarity functions beyond cosine

### 🚀 New Features
- [ ] Implement the database-driven architecture from REDESIGN.md
- [ ] Add graph analysis functions (centrality, clustering, etc.)
- [ ] Support for incremental updates (add new nodes without recomputing all embeddings)
- [ ] Web UI for interactive exploration

## Code Quality

### 🧹 Cleanup Tasks
- [ ] Remove unused imports across all modules
- [ ] Add type hints to remaining functions
- [ ] Standardize logging across all modules
- [ ] Add docstring examples to all public API methods

### 🔍 Code Review Items
- [ ] Review error handling patterns - ensure consistent user-friendly messages
- [ ] Audit security - ensure no secrets can be logged
- [ ] Performance review - identify bottlenecks in large graph processing
- [ ] Memory usage optimization for large embedding matrices

## Infrastructure

### 📦 Packaging & Distribution
- [ ] Set up GitHub Actions for CI/CD
- [ ] Add pre-commit hooks for code formatting
- [ ] Package for PyPI distribution
- [ ] Create Docker containers for deployment

### 🔧 Developer Experience
- [ ] Add debugging utilities for troubleshooting embeddings
- [ ] Create development setup script
- [ ] Add benchmarking tools
- [ ] Improve error messages with suggestions

## Future Architecture Considerations

### 🏗️ Scalability
- [ ] Implement chunked processing for very large datasets
- [ ] Add distributed computing support (Dask/Ray)
- [ ] Consider graph database backends (Neo4j, ArangoDB)
- [ ] Streaming processing for real-time updates

### 🔌 Integrations
- [ ] Add more data source integrations (Slack, Discord, etc.)
- [ ] Support for multiple embedding models simultaneously
- [ ] Integration with vector databases (Pinecone, Weaviate, Chroma)
- [ ] Export to more graph formats (Cypher, GraphSON)

### 🧠 Advanced Features
- [ ] Graph neural networks for node classification
- [ ] Temporal analysis for evolving networks
- [ ] Multi-modal embeddings (text + images)
- [ ] Hierarchical clustering and community detection

## Completed ✅

- ✅ Refactor CLI to use clean API architecture
- ✅ Separate business logic from I/O operations
- ✅ Create comprehensive test suite with good coverage
- ✅ Implement fluent API for complex workflows
- ✅ Clean up legacy code and improve maintainability
- ✅ Achieve 99% CLI coverage and 56% overall coverage

---

**Priority Order:**
1. Fix failing test and documentation updates (immediate)
2. Add integration tests and API enhancements (short-term)
3. New features and infrastructure improvements (medium-term)
4. Advanced scalability and ML features (long-term)

**Note:** This TODO list reflects the current state after the major CLI refactoring. Focus on stabilizing the new architecture before adding new features.