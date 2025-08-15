# Workdir Configuration Guide

## Overview

AGraph API now uses a centralized `workdir` configuration to organize all persistent data
storage directories. This ensures better organization and makes it easier to manage data
locations.

## Directory Structure

All AGraph data is now stored under the configured `workdir` directory:

```text
workdir/
├── agraph_vectordb/          # Vector database storage (Chroma)
│   └── chroma/
├── document_storage/         # Uploaded documents storage
│   ├── documents/           # Raw document content
│   ├── metadata/            # Document metadata
│   └── index.json           # Document index
└── cache/                   # Knowledge graph build cache
    ├── metadata/
    ├── steps/
    └── user_edits/
```

## Configuration

### Default Configuration

The default `workdir` is set to `"workdir"` in the current directory:

```python
# agraph/config.py
class Settings(BaseModel):
    workdir: str = Field(default="workdir")
```

### Environment Variable Configuration

You can override the workdir using the `AGRAPH_WORKDIR` environment variable:

```bash
# Set custom workdir
export AGRAPH_WORKDIR="/path/to/your/custom/workdir"

# Or in .env file
echo "AGRAPH_WORKDIR=/path/to/your/custom/workdir" >> .env
```

### Programmatic Configuration

You can also set it programmatically when creating settings:

```python
from agraph.config import Settings

# Create settings with custom workdir
settings = Settings(workdir="/path/to/custom/workdir")
```

## Benefits

### 1. **Centralized Storage**

- All AGraph data is organized under a single directory
- Easier backup and migration of all data
- Clear separation from other project files

### 2. **Portable Deployment**

- Easy to move entire AGraph installation by copying workdir
- Consistent paths across different environments
- Simplified Docker volume mounting

### 3. **Better Organization**

- Logical separation of different data types
- No scattered directories in the project root
- Easier to clean up or reset specific components

## Usage Examples

### Basic Usage

```python
from agraph.api.dependencies import get_agraph_instance, get_document_manager

# All paths will be under workdir automatically
agraph = await get_agraph_instance()
doc_manager = get_document_manager()

# Check paths
print(f"Vector DB: {agraph.persist_directory}")         # workdir/agraph_vectordb
print(f"Documents: {doc_manager.storage_dir}")          # workdir/document_storage
print(f"Cache: {agraph.config.cache_dir}")              # workdir/cache
```

### Docker Configuration

```dockerfile
# Dockerfile
ENV AGRAPH_WORKDIR=/app/data
VOLUME ["/app/data"]
```

```yaml
# docker-compose.yml
services:
  agraph-api:
    environment:
      - AGRAPH_WORKDIR=/app/data
    volumes:
      - ./agraph-data:/app/data
```

### Multiple Environment Setup

```bash
# Development
export AGRAPH_WORKDIR="./dev-workdir"

# Testing
export AGRAPH_WORKDIR="./test-workdir"

# Production
export AGRAPH_WORKDIR="/var/lib/agraph"
```

## Migration Guide

### From Previous Versions

If you have existing data in the old locations:

```bash
# Create new workdir structure
mkdir -p workdir

# Move existing data (if it exists)
[ -d "./agraph_vectordb" ] && mv ./agraph_vectordb workdir/
[ -d "./document_storage" ] && mv ./document_storage workdir/
[ -d "./cache" ] && mv ./cache workdir/

# Clean up old scattered directories
rm -rf ./agraph_vectordb ./document_storage ./cache
```

### Automatic Migration

The system will automatically create the required directories when they don't exist:

```python
# These calls will create workdir/agraph_vectordb, workdir/document_storage, etc.
agraph = await get_agraph_instance()
doc_manager = get_document_manager()
```

## Advanced Configuration

### Custom Path Components

You can further customize individual component paths by modifying the dependencies:

```python
# agraph/api/dependencies.py (example customization)
async def get_agraph_instance() -> AGraph:
    settings = get_settings()

    # Custom subdirectory for vectordb
    custom_vectordb_path = f"{settings.workdir}/custom/vectordb"

    _agraph_instance = AGraph(
        persist_directory=custom_vectordb_path,
        # ... other config
    )
```

### Multiple Workdirs

For advanced use cases, you might want different components in different locations:

```python
# Different workdirs for different components
vector_workdir = os.getenv("AGRAPH_VECTOR_WORKDIR", settings.workdir)
doc_workdir = os.getenv("AGRAPH_DOC_WORKDIR", settings.workdir)

vectordb_path = f"{vector_workdir}/agraph_vectordb"
document_path = f"{doc_workdir}/document_storage"
```

## Monitoring and Maintenance

### Check Disk Usage

```bash
# Check total workdir size
du -sh workdir

# Check individual components
du -sh workdir/agraph_vectordb
du -sh workdir/document_storage
du -sh workdir/cache
```

### Backup Strategy

```bash
# Simple backup
tar -czf agraph-backup-$(date +%Y%m%d).tar.gz workdir/

# Incremental backup (rsync)
rsync -av workdir/ /backup/location/workdir/
```

### Clean Up

```bash
# Clean cache only
rm -rf workdir/cache

# Reset everything (careful!)
rm -rf workdir
```

## Troubleshooting

### Permission Issues

```bash
# Ensure proper permissions
chmod -R 755 workdir
chown -R $(whoami):$(whoami) workdir
```

### Path Not Found Errors

1. Check environment variable: `echo $AGRAPH_WORKDIR`
2. Verify directory permissions
3. Check if parent directory exists and is writable

### Migration Issues

1. Stop all AGraph services first
2. Move data directories carefully
3. Update environment variables
4. Restart services and verify paths

## Security Considerations

- Ensure workdir has appropriate file permissions
- Don't store workdir in publicly accessible locations
- Consider encryption for sensitive document storage
- Regular backups of the entire workdir
