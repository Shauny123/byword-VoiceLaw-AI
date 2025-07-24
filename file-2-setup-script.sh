#!/bin/bash
# setup_complete_legal_intake.sh
# Complete setup for Legal AI Intake System based on PodGPT

set -e  # Exit on any error

echo "🏗️  Setting up Complete Legal AI Intake System"
echo "📋 Based on PodGPT's RAG architecture with legal domain adaptation"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
print_step "Checking system requirements..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3.8+ required but not found. Please install Python first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_success "Python $PYTHON_VERSION found"

# Check Git
if ! command -v git &> /dev/null; then
    print_error "Git required but not found. Please install Git first."
    exit 1
fi

print_success "Git found"

# Check Node.js for frontend
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Node.js $NODE_VERSION found"
    HAS_NODE=true
else
    print_warning "Node.js not found - frontend features will be limited"
    HAS_NODE=false
fi

# Check if we have GPU support
if command -v nvidia-smi &> /dev/null; then
    print_success "🚀 NVIDIA GPU detected - will use GPU acceleration"
    USE_GPU=true
else
    print_warning "No GPU detected - will use CPU (slower but functional)"
    USE_GPU=false
fi

# Step 1: Create project directory and virtual environment
print_step "1/8 Setting up project environment..."

if [ -d "legal-ai-intake-env" ]; then
    print_warning "Environment already exists, activating..."
    source legal-ai-intake-env/bin/activate
else
    python3 -m venv legal-ai-intake-env
    source legal-ai-intake-env/bin/activate
    print_success "Virtual environment created and activated"
fi

# Step 2: Clone PodGPT and convert to legal
print_step "2/8 Cloning PodGPT and converting to legal system..."

if [ ! -f "convert_podgpt_legal.py" ]; then
    print_error "convert_podgpt_legal.py not found. Please ensure the conversion script is in the current directory."
    exit 1
fi

python3 convert_podgpt_legal.py
print_success "PodGPT successfully converted to Legal AI Intake System"

# Step 3: Install Python requirements
print_step "3/8 Installing Python dependencies..."

cd ai-legal-intake

# Install base requirements
pip install --upgrade pip
pip install -r requirements.txt

# Install additional legal-specific packages
pip install \
    python-multipart \
    python-jose[cryptography] \
    passlib[bcrypt] \
    python-docx \
    PyPDF2 \
    phonemizer \
    espeak-ng

if [ "$USE_GPU" = true ]; then
    print_step "Installing GPU-accelerated packages..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install faiss-gpu
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install faiss-cpu
fi

print_success "All Python dependencies installed"

# Step 4: Download language models
print_step "4/8 Downloading language models..."

# Download spaCy models for multilingual support
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download de_core_news_sm  
python -m spacy download fr_core_news_sm
python -m spacy download zh_core_web_sm

print_success "Language models downloaded"

# Step 5: Download AI models
print_step "5/8 Downloading AI models for legal processing..."

# Create models directory
mkdir -p models

# Download legal-specific models
python scripts/download_model.py

print_success "AI models downloaded"

# Step 6: Setup legal datasets
print_step "6/8 Setting up legal datasets..."

# Create datasets directory structure
mkdir -p datasets/{legal_qa,case_law,statutes,multilingual}

# Create sample legal dataset
cat > datasets/legal_qa/sample_legal_qa.jsonl << 'EOF'
{"question": "I was in a car accident and the other driver was texting. What should I do?", "answer": "First, ensure everyone is safe and call 911 if there are injuries. Document the scene with photos, get the other driver's insurance information, and file a police report. The fact that the other driver was texting could establish negligence. Contact a personal injury attorney to discuss your case.", "domain": "personal_injury", "language": "en"}
{"question": "My employer fired me after I complained about harassment. Is this legal?", "answer": "This could be wrongful termination and retaliation, which are illegal under federal employment laws. Document all incidents of harassment and your complaints. File a complaint with the EEOC within 180 days. You may have grounds for a lawsuit against your employer.", "domain": "employment_law", "language": "en"}
{"question": "¿Qué debo hacer si mi cónyuge quiere el divorcio?", "answer": "Debe consultar con un abogado de derecho familiar para entender sus derechos. Reúna documentos financieros, considere la mediación, y piense en la custodia de los hijos si los hay. Las leyes de divorcio varían por estado.", "domain": "family_law", "language": "es"}
{"question": "我被警察逮捕了，我应该怎么办？", "answer": "行使你的沉默权，不要在没有律师在场的情况下回答问题。要求立即联系律师。不要签署任何文件或承认任何事情。记住，你有权获得法律代表。", "domain": "criminal_law", "language": "zh"}
EOF

# Create sample case law data
cat > datasets/case_law/sample_cases.jsonl << 'EOF'
{"case_name": "Brown v. Board of Education", "year": 1954, "court": "Supreme Court", "summary": "Landmark case that declared racial segregation in public schools unconstitutional", "outcome": "Unanimous decision ending legal segregation", "legal_principle": "Equal protection under the law", "domain": "civil_rights"}
{"case_name": "Miranda v. Arizona", "year": 1966, "court": "Supreme Court", "summary": "Established requirement for police to inform suspects of their rights", "outcome": "Created Miranda rights requirement", "legal_principle": "Fifth Amendment protection against self-incrimination", "domain": "criminal_law"}
EOF

# Create sample statutes
cat > datasets/statutes/sample_statutes.jsonl << 'EOF'
{"statute": "Civil Rights Act of 1964 - Title VII", "section": "42 U.S.C. §2000e", "summary": "Prohibits employment discrimination based on race, color, religion, sex, or national origin", "jurisdiction": "federal", "domain": "employment_law"}
{"statute": "Americans with Disabilities Act", "section": "42 U.S.C. §12101", "summary": "Prohibits discrimination against individuals with disabilities in employment, public accommodations, and other areas", "jurisdiction": "federal", "domain": "civil_rights"}
EOF

print_success "Sample legal datasets created"

# Step 7: Configure environment variables
print_step "7/8 Setting up environment configuration..."

# Create .env file with default values
cat > .env << 'EOF'
# AI Legal Intake System Configuration

# Application Settings
APP_NAME="AI Legal Intake System"
APP_VERSION="1.0.0"
DEBUG=false
PORT=8000

# API Keys (Add your keys here)
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_TRANSLATE_API_KEY=your_google_translate_key_here
AZURE_TRANSLATOR_KEY=your_azure_translator_key_here
NVIDIA_API_KEY=your_nvidia_api_key_here

# Database (optional)
DATABASE_URL=sqlite:///./legal_intake.db

# Feature Flags
ENABLE_VOICE_PROCESSING=true
ENABLE_REAL_TIME_TRANSLATION=true
ENABLE_MULTILINGUAL_SUPPORT=true
ENABLE_COST_OPTIMIZATION=true

# Legal Settings
REQUIRE_LEGAL_DISCLAIMER=true
DATA_RETENTION_DAYS=30
CONFIDENTIALITY_MODE=true

# Performance Settings
MAX_CONCURRENT_SESSIONS=100
RESPONSE_TIMEOUT_SECONDS=30
AUDIO_MAX_SIZE_MB=10
EOF

print_success "Environment configuration created"

# Step 8: Setup frontend (if Node.js available)
if [ "$HAS_NODE" = true ]; then
    print_step "8/8 Setting up frontend components..."
    
    # Create frontend directory
    mkdir -p frontend/src/{components,hooks,api,utils}
    
    # Initialize package.json
    cat > frontend/package.json << 'EOF'
{
  "name": "legal-ai-intake-frontend",
  "version": "1.0.0",
  "description": "Frontend for Legal AI Intake System with Advanced Lip Sync",
  "main": "src/index.tsx",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "^14.0.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "typescript": "^5.0.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "tailwindcss": "^3.3.0",
    "lucide-react": "^0.263.1",
    "framer-motion": "^10.16.0",
    "socket.io-client": "^4.7.0",
    "@types/node": "^20.0.0"
  },
  "devDependencies": {
    "eslint": "^8.45.0",
    "eslint-config-next": "^14.0.0"
  }
}
EOF
    
    cd frontend
    npm install
    cd ..
    
    print_success "Frontend dependencies installed"
else
    print_step "8/8 Skipping frontend setup (Node.js not available)"
fi

# Create startup scripts
print_step "Creating startup scripts..."

# Create development server script
cat > start_dev_server.sh << 'EOF'
#!/bin/bash
# Development server startup script

echo "🚀 Starting Legal AI Intake System (Development Mode)"

# Activate virtual environment
source ../legal-ai-intake-env/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Start the FastAPI server
echo "📡 Starting API server on http://localhost:8000"
python src/api/natural_voice_backend.py &

# Start frontend if available
if [ -d "frontend" ] && command -v npm &> /dev/null; then
    echo "🌐 Starting frontend on http://localhost:3000"
    cd frontend && npm run dev &
    cd ..
fi

echo "✅ Servers started!"
echo "📱 Frontend: http://localhost:3000 (if available)"
echo "🔌 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for processes
wait
EOF

chmod +x start_dev_server.sh

# Create production server script
cat > start_production.sh << 'EOF'
#!/bin/bash
# Production server startup script

echo "🏭 Starting Legal AI Intake System (Production Mode)"

# Activate virtual environment
source ../legal-ai-intake-env/bin/activate

# Set production environment
export ENVIRONMENT=production
export DEBUG=false

# Start with Gunicorn for production
pip install gunicorn

gunicorn src.api.natural_voice_backend:app \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --access-logfile access.log \
    --error-logfile error.log \
    --daemon

echo "✅ Production server started on port 8000"
echo "📊 Logs: access.log, error.log"
EOF

chmod +x start_production.sh

# Create test script
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""Test Legal AI Intake System installation"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported"""
    required_packages = [
        'torch', 'transformers', 'sentence_transformers', 
        'faiss', 'openai', 'whisper', 'spacy', 'fastapi',
        'uvicorn', 'requests', 'numpy', 'pandas', 'pydub'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def test_models():
    """Test that model directories exist"""
    models_dir = Path("./models")
    if not models_dir.exists():
        print("❌ Models directory not found")
        return False
    
    print("✅ Models directory exists")
    return True

def test_datasets():
    """Test that dataset directories exist"""
    datasets_dir = Path("./datasets")
    if not datasets_dir.exists():
        print("❌ Datasets directory not found")
        return False
    
    print("✅ Datasets directory exists")
    return True

def test_config():
    """Test that configuration files exist"""
    config_files = [
        "./config/legal_config.json",
        "./.env",
        "./requirements.txt"
    ]
    
    all_exist = True
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} not found")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🧪 Testing Legal AI Intake System Installation")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Models Setup", test_models), 
        ("Datasets Setup", test_datasets),
        ("Configuration", test_config)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}:")
        if not test_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Your Legal AI Intake System is ready!")
        print("\n🚀 Quick Start:")
        print("1. Add your API keys to .env file")
        print("2. Run: ./start_dev_server.sh")
        print("3. Open: http://localhost:8000/docs")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x test_installation.py

print_success "Startup scripts created"

# Final summary
cd ..

echo ""
echo "🎉 LEGAL AI INTAKE SYSTEM SETUP COMPLETED!"
echo "================================================"
echo ""
echo "📁 Project Location: ./ai-legal-intake/"
echo "🌐 Based on PodGPT's excellent RAG architecture"
echo ""
echo "🚀 Quick Start:"
echo "1. cd ai-legal-intake"
echo "2. Edit .env file with your API keys"
echo "3. ./start_dev_server.sh"
echo ""
echo "🌍 Features:"
echo "- 30+ language support with voice processing"
echo "- Legal entity extraction and case classification" 
echo "- Cultural legal adaptations per jurisdiction"
echo "- Cost-optimized audio processing (Whisper + NVIDIA Flamingo 3)"
echo "- Real-time translation and conversation"
echo "- Advanced lip sync with natural turn-taking"
echo ""
echo "📚 Documentation:"
echo "- README.md - Complete system overview"
echo "- CONVERSION_SUMMARY.json - Details of PodGPT conversion"
echo "- config/legal_config.json - System configuration"
echo ""
echo "🎯 Next Steps:"
echo "1. Get API keys: OpenAI, Google Translate, NVIDIA"
echo "2. Test with: python test_installation.py"
echo "3. Start development: ./start_dev_server.sh"
echo "4. Deploy production: ./start_production.sh"
echo ""
echo "Built with ❤️  on PodGPT's solid foundation"
echo "Ready to serve 5.5+ billion people worldwide! 🌎"
