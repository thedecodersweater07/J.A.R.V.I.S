#!/bin/bash

# Move core files
mv core/brain/* core/brain/cognitive/
mv core/memory/* core/memory/long_term/

# Move UI files
mv ui/screen.py ui/screens/base/
mv ui/visual/* ui/visual/renderers/

# Move LLM files
mv llm/core/* llm/models/architecture/
mv llm/memory/* llm/memory/enhanced/

# Move database files
mv db/database.py db/sql/models/
mv db/database_manager.py db/sql/

# Move config files
mv config/*.json config/defaults/

# Create symlinks for backwards compatibility
ln -s core/brain/cognitive/cerebrum.py core/brain/cerebrum.py
ln -s ui/screens/base/screen.py ui/screen.py
ln -s db/sql/database.py db/database.py
