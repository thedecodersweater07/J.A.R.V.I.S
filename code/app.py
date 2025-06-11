#!/usr/bin/env python3
"""
Python Code Analyzer Backend Server
Full-stack applicatie voor code analyse en automatische fixes
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import ast
import sys
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import traceback
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
from datetime import datetime
import threading
import zipfile
from werkzeug.utils import secure_filename
import logging 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Flask applicatie setup

app = Flask(__name__)
CORS(app)

# Configuratie
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'py', 'zip', 'tar', 'gz'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Zorg ervoor dat directories bestaan
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

class ProblemType(Enum):
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    UNDEFINED_VARIABLE = "undefined_variable"
    UNUSED_IMPORT = "unused_import"
    INDENTATION_ERROR = "indentation_error"
    NAME_ERROR = "name_error"
    TYPE_ERROR = "type_error"
    ATTRIBUTE_ERROR = "attribute_error"
    MISSING_DEPENDENCY = "missing_dependency"
    DEPRECATED_SYNTAX = "deprecated_syntax"
    CODE_STYLE = "code_style"

class LOG_LEVEL(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
def allowed_file(filename: str) -> bool:
    """Controleer of bestandsextensie is toegestaan"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def is_valid_file_size(file_path: str) -> bool:
    """Controleer of bestand niet te groot is"""
    return os.path.getsize(file_path) <= MAX_FILE_SIZE
@dataclass
class CodeProblem:
    file_path: str
    line_number: int
    problem_type: str
    description: str
    suggestion: str
    original_code: str = ""
    fixed_code: str = ""
    severity: str = "medium"  # low, medium, high, critical

class AdvancedCodeAnalyzer:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.problems: List[CodeProblem] = []
        self.python_files: List[Path] = []
        self.analysis_id = str(uuid.uuid4())
        self.progress = 0
        
    def analyze_codebase(self) -> Dict:
        """Analyseer volledige codebase"""
        self.discover_python_files()
        total_files = len(self.python_files)
        
        results = {
            'analysis_id': self.analysis_id,
            'timestamp': datetime.now().isoformat(),
            'total_files': total_files,
            'problems': [],
            'summary': {},
            'fixes_applied': []
        }
        
        for i, file_path in enumerate(self.python_files):
            self.progress = int((i / total_files) * 100)
            file_problems = self.analyze_file(file_path)
            self.problems.extend(file_problems)
            results['problems'].extend([asdict(p) for p in file_problems])
            print(f"Analyzing {file_path}... {self.progress}% complete")
            if self.progress % 10 == 0:
                print(f"Progress: {self.progress}% - {file_path.name} analyzed")
        # Sla resultaten op in een bestand
            
        # Converteer problems naar dictionary format
        results['problems'] = [asdict(p) for p in self.problems]
        results['summary'] = self.generate_summary()
        
        return results
    
    def discover_python_files(self) -> List[Path]:
        """Ontdek alle Python bestanden"""
        python_files = []
        for file_path in self.root_path.rglob("*.py"):
            if not any(part.startswith('.') for part in file_path.parts):
                if 'venv' not in str(file_path) and '__pycache__' not in str(file_path):
                    python_files.append(file_path)
        self.python_files = python_files
        return python_files
    
    def analyze_file(self, file_path: Path) -> List[CodeProblem]:
        """Analyseer een enkel bestand"""
        problems = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Verschillende analyses
            problems.extend(self._check_syntax(file_path, content))
            problems.extend(self._check_imports(file_path, content))
            problems.extend(self._check_variables(file_path, content))
            problems.extend(self._check_code_style(file_path, content))
            problems.extend(self._check_deprecated_syntax(file_path, content))
            problems.extend(self._check_common_errors(file_path, content))
            
        except Exception as e:
            problems.append(CodeProblem(
                file_path=str(file_path),
                line_number=1,
                problem_type=ProblemType.SYNTAX_ERROR.value,
                description=f"Kan bestand niet analyseren: {str(e)}",
                suggestion="Controleer bestandspermissies en encoding",
                severity="high"
            ))
            
        return problems
    
    def _check_syntax(self, file_path: Path, content: str) -> List[CodeProblem]:
        """Check syntax errors"""
        problems = []
        
        try:
            ast.parse(content)
        except SyntaxError as e:
            fixed_code = self._auto_fix_syntax(content, e)
            problems.append(CodeProblem(
                file_path=str(file_path),
                line_number=e.lineno or 1,
                problem_type=ProblemType.SYNTAX_ERROR.value,
                description=f"Syntax fout: {e.msg}",
                suggestion=self._suggest_syntax_fix(e),
                original_code=e.text or "",
                fixed_code=fixed_code,
                severity="critical"
            ))
        except IndentationError as e:
            fixed_code = self._auto_fix_indentation(content)
            problems.append(CodeProblem(
                file_path=str(file_path),
                line_number=e.lineno or 1,
                problem_type=ProblemType.INDENTATION_ERROR.value,
                description=f"Indentatie fout: {e.msg}",
                suggestion="Automatisch gefixte indentatie beschikbaar",
                original_code=e.text or "",
                fixed_code=fixed_code,
                severity="high"
            ))
            
        return problems
    
    def _check_imports(self, file_path: Path, content: str) -> List[CodeProblem]:
        """Check import problemen"""
        problems = []
        
        try:
            tree = ast.parse(content)
        except:
            return problems
            
        imports = []
        used_names = set()
        
        # Verzamel imports en usage
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_name = alias.asname or alias.name
                    imports.append((import_name, alias.name, node.lineno))
                    
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    import_name = alias.asname or alias.name
                    imports.append((import_name, f"{module}.{alias.name}", node.lineno))
                    
            elif isinstance(node, ast.Name):
                used_names.add(node.id)
        
        # Check ongebruikte imports
        for import_name, full_name, line_no in imports:
            if import_name not in used_names and import_name != "*":
                fixed_content = self._remove_unused_import(content, line_no)
                problems.append(CodeProblem(
                    file_path=str(file_path),
                    line_number=line_no,
                    problem_type=ProblemType.UNUSED_IMPORT.value,
                    description=f"Ongebruikte import: {import_name}",
                    suggestion="Import automatisch verwijderd",
                    fixed_code=fixed_content,
                    severity="low"
                ))
        
        return problems
    
    def _check_code_style(self, file_path: Path, content: str) -> List[CodeProblem]:
        """Check code style problemen"""
        problems = []
        lines = content.split('\n')
        
        for line_no, line in enumerate(lines, 1):
            # Lange lijnen
            if len(line) > 79:
                problems.append(CodeProblem(
                    file_path=str(file_path),
                    line_number=line_no,
                    problem_type=ProblemType.CODE_STYLE.value,
                    description=f"Lijn te lang ({len(line)} karakters)",
                    suggestion="Splits lange lijnen op volgens PEP 8",
                    original_code=line.strip(),
                    severity="low"
                ))
            
            # Trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                fixed_line = line.rstrip()
                problems.append(CodeProblem(
                    file_path=str(file_path),
                    line_number=line_no,
                    problem_type=ProblemType.CODE_STYLE.value,
                    description="Trailing whitespace",
                    suggestion="Whitespace verwijderd",
                    original_code=line,
                    fixed_code=fixed_line,
                    severity="low"
                ))
        
        return problems
    
    def _check_deprecated_syntax(self, file_path: Path, content: str) -> List[CodeProblem]:
        """Check deprecated syntax"""
        problems = []
        lines = content.split('\n')
        
        deprecated_patterns = [
            (r'string\.', "Gebruik str methods in plaats van string module", "medium"),
            (r'\.has_key\(', "Gebruik 'in' operator in plaats van has_key()", "medium"),
            (r'print\s+[^(]', "Gebruik print() functie syntax", "high"),
            (r'xrange\(', "Gebruik range() in plaats van xrange()", "high"),
        ]
        
        for line_no, line in enumerate(lines, 1):
            for pattern, suggestion, severity in deprecated_patterns:
                if re.search(pattern, line):
                    fixed_line = self._fix_deprecated_syntax(line, pattern)
                    problems.append(CodeProblem(
                        file_path=str(file_path),
                        line_number=line_no,
                        problem_type=ProblemType.DEPRECATED_SYNTAX.value,
                        description="Deprecated syntax gevonden",
                        suggestion=suggestion,
                        original_code=line.strip(),
                        fixed_code=fixed_line,
                        severity=severity
                    ))
        
        return problems
    
    def _check_common_errors(self, file_path: Path, content: str) -> List[CodeProblem]:
        """Check voor veel voorkomende fouten"""
        problems = []
        lines = content.split('\n')
        
        common_issues = [
            (r'=\s*=', "Mogelijk == in plaats van = bedoeld", "medium"),
            (r'if\s+.*=\s*[^=]', "Assignment in if statement", "high"),
            (r'except:', "Bare except clause - specificeer exception type", "medium"),
        ]
        
        for line_no, line in enumerate(lines, 1):
            for pattern, description, severity in common_issues:
                if re.search(pattern, line):
                    problems.append(CodeProblem(
                        file_path=str(file_path),
                        line_number=line_no,
                        problem_type=ProblemType.CODE_STYLE.value,
                        description=description,
                        suggestion="Review en corrigeer indien nodig",
                        original_code=line.strip(),
                        severity=severity
                    ))
        
        return problems
    
    def _auto_fix_syntax(self, content: str, error: SyntaxError) -> str:
        """Probeer syntax fouten automatisch te fixen"""
        lines = content.split('\n')
        if not error.lineno or error.lineno > len(lines):
            return content
            
        line_idx = error.lineno - 1
        line = lines[line_idx]
        
        # Simpele fixes
        if "invalid syntax" in (error.msg or "").lower():
            # Fix missing colons
            if re.match(r'^\s*(if|for|while|def|class|try|except|with)\s+.*[^:]$', line):
                lines[line_idx] = line + ':'
            # Fix unclosed parentheses
            elif line.count('(') > line.count(')'):
                lines[line_idx] = line + ')'
            elif line.count('[') > line.count(']'):
                lines[line_idx] = line + ']'
                
        return '\n'.join(lines)
    
    def _auto_fix_indentation(self, content: str) -> str:
        """Fix indentation problemen"""
        lines = content.split('\n')
        fixed_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                fixed_lines.append('')
                continue
                
            # Bepaal indent level
            if stripped.endswith(':'):
                fixed_lines.append('    ' * indent_level + stripped)
                indent_level += 1
            elif stripped.startswith(('return', 'break', 'continue', 'pass')):
                fixed_lines.append('    ' * indent_level + stripped)
            elif stripped.startswith(('except', 'elif', 'else', 'finally')):
                indent_level = max(0, indent_level - 1)
                fixed_lines.append('    ' * indent_level + stripped)
                indent_level += 1
            else:
                fixed_lines.append('    ' * indent_level + stripped)
                
        return '\n'.join(fixed_lines)
    
    def _remove_unused_import(self, content: str, line_no: int) -> str:
        """Verwijder ongebruikte import"""
        lines = content.split('\n')
        if 1 <= line_no <= len(lines):
            lines[line_no - 1] = ''  # Lege lijn in plaats van verwijderen
        return '\n'.join(lines)
    
    def _fix_deprecated_syntax(self, line: str, pattern: str) -> str:
        """Fix deprecated syntax"""
        if r'print\s+[^(]' in pattern:
            # Fix print statement naar print functie
            match = re.search(r'print\s+(.+)', line)
            if match:
                content = match.group(1)
                return line.replace(match.group(0), f'print({content})')
        elif r'xrange\(' in pattern:
            return line.replace('xrange(', 'range(')
        elif r'\.has_key\(' in pattern:
            # Fix dict.has_key() naar 'key' in dict
            return re.sub(r'(\w+)\.has_key\(([^)]+)\)', r'\2 in \1', line)
            
        return line
    
    def _suggest_syntax_fix(self, error: SyntaxError) -> str:
        """Suggereer syntax fix"""
        msg = (error.msg or "").lower()
        
        if "invalid syntax" in msg:
            return "Controleer ontbrekende dubbele punten, haakjes of quotes"
        elif "unexpected eof" in msg:
            return "Bestand eindigt onverwacht - controleer ontbrekende haakjes"
        elif "unmatched" in msg:
            return "Niet-matchende haakjes of quotes"
        else:
            return "Controleer syntax rond deze lijn"
    
    def generate_summary(self) -> Dict:
        """Genereer samenvatting van problemen"""
        summary = {
            'total_problems': len(self.problems),
            'by_type': {},
            'by_severity': {},
            'fixable_problems': 0,
            'files_with_problems': len(set(p.file_path for p in self.problems))
        }
        
        for problem in self.problems:
            # By type
            p_type = problem.problem_type
            summary['by_type'][p_type] = summary['by_type'].get(p_type, 0) + 1
            
            # By severity  
            severity = problem.severity
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            # Fixable
            if problem.fixed_code:
                summary['fixable_problems'] += 1
                
        return summary
    
    def apply_fixes(self, selected_problems: List[str]) -> Dict:
        """Pas automatische fixes toe"""
        results = {
            'fixes_applied': 0,
            'files_modified': set(),
            'errors': []
        }
        
        # Group problems by file
        file_problems = {}
        for i, problem in enumerate(self.problems):
            if str(i) in selected_problems and problem.fixed_code:
                file_path = problem.file_path
                if file_path not in file_problems:
                    file_problems[file_path] = []
                file_problems[file_path].append(problem)
        
        # Apply fixes per file
        for file_path, problems in file_problems.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Apply fixes (simplified - in reality more complex)
                for problem in problems:
                    if problem.original_code and problem.fixed_code:
                        content = content.replace(problem.original_code, problem.fixed_code)
                
                # Backup original
                backup_path = f"{file_path}.backup"
                shutil.copy(file_path, backup_path)
                
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                results['fixes_applied'] += len(problems)
                results['files_modified'].add(file_path)
                
            except Exception as e:
                results['errors'].append(f"Error fixing {file_path}: {str(e)}")
        
        results['files_modified'] = list(results['files_modified'])
        return results

# Global storage voor analyses
analyses = {}

@app.route('/')
def index():
    """Serve de frontend"""
    return send_from_directory('.', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Upload en analyseer bestanden"""
    if 'files' not in request.files:
        return jsonify({'error': 'Geen bestanden gevonden'}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'Geen bestanden geselecteerd'}), 400
    
    # Create unique directory voor deze upload
    upload_id = str(uuid.uuid4())
    upload_dir = os.path.join(UPLOAD_FOLDER, upload_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    try:
        # Save uploaded files
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(upload_dir, filename)
                file.save(file_path)
                
                # Extract if zip
                if filename.endswith('.zip'):
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(upload_dir)
        
        # Start analysis in background
        analyzer = AdvancedCodeAnalyzer(upload_dir)
        
        def run_analysis():
            results = analyzer.analyze_codebase()
            analyses[upload_id] = results
        
        thread = threading.Thread(target=run_analysis)
        thread.start()
        
        return jsonify({
            'upload_id': upload_id,
            'message': 'Bestanden geüpload, analyse gestart'
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload error: {str(e)}'}), 500

@app.route('/analyze/<upload_id>')
def get_analysis(upload_id):
    """Haal analyse resultaten op"""
    if upload_id not in analyses:
        return jsonify({'status': 'analyzing', 'progress': 0})
    
    return jsonify(analyses[upload_id])

@app.route('/fix', methods=['POST'])
def apply_fixes():
    """Pas automatische fixes toe"""
    data = request.get_json()
    upload_id = data.get('upload_id')
    selected_problems = data.get('selected_problems', [])
    
    if upload_id not in analyses:
        return jsonify({'error': 'Analyse niet gevonden'}), 404
    
    upload_dir = os.path.join(UPLOAD_FOLDER, upload_id)
    analyzer = AdvancedCodeAnalyzer(upload_dir)
    analyzer.problems = [CodeProblem(**p) for p in analyses[upload_id]['problems']]
    
    results = analyzer.apply_fixes(selected_problems)
    return jsonify(results)

@app.route('/generate', methods=['POST'])
def generate_code():
    """Genereer nieuwe Python code"""
    data = request.get_json()
    description = data.get('description', '')
    code_type = data.get('type', 'function')  # function, class, script
    
    # AI-style code generation (simplified)
    generated_code = generate_python_code(description, code_type)
    
    return jsonify({
        'generated_code': generated_code,
        'filename': f'generated_{uuid.uuid4().hex[:8]}.py'
    })

def generate_python_code(description: str, code_type: str) -> str:
    """Genereer Python code gebaseerd op beschrijving"""
    
    templates = {
        'function': '''def {name}({params}):
    """
    {description}
    """
    # TODO: Implementeer functionaliteit
    pass
''',
        'class': '''class {name}:
    """
    {description}
    """
    
    def __init__(self{params}):
        # TODO: Initialiseer attributen
        pass
    
    def __str__(self):
        return f"{name} instance"
''',
        'script': '''#!/usr/bin/env python3
"""
{description}
"""

def main():
    """Main functie"""
    # TODO: Implementeer hoofdlogica
    pass

if __name__ == "__main__":
    main()
'''
    }
    
    # Extract name en parameters uit description (simplified)
    import re
    
    name_match = re.search(r'(?:functie|class|script)\s+(\w+)', description.lower())
    name = name_match.group(1) if name_match else 'generated_item'
    
    params_match = re.search(r'parameters?\s*:?\s*([^.]+)', description.lower())
    params = ', ' + params_match.group(1).strip() if params_match else ''
    
    template = templates.get(code_type, templates['function'])
    
    return template.format(
        name=name,
        params=params,
        description=description
    )

@app.route('/debug', methods=['POST'])
def debug_code():
    """Debug Python code"""
    data = request.get_json()
    code = data.get('code', '')
    
    debug_results = {
        'syntax_valid': True,
        'issues': [],
        'suggestions': [],
        'execution_test': None
    }
    
    # Syntax check
    try:
        ast.parse(code)
    except SyntaxError as e:
        debug_results['syntax_valid'] = False
        debug_results['issues'].append({
            'type': 'syntax_error',
            'line': e.lineno,
            'message': e.msg,
            'text': e.text
        })
    
    # Probeer code uit te voeren in safe environment
    if debug_results['syntax_valid']:
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Run with timeout
            result = subprocess.run([
                sys.executable, temp_file
            ], capture_output=True, text=True, timeout=5)
            
            debug_results['execution_test'] = {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            os.unlink(temp_file)
            
        except subprocess.TimeoutExpired:
            debug_results['execution_test'] = {
                'success': False,
                'error': 'Code execution timeout'
            }
        except Exception as e:
            debug_results['execution_test'] = {
                'success': False,
                'error': str(e)
            }
    
    return jsonify(debug_results)

if __name__ == '__main__':
    print("🚀 Python Code Analyzer Backend gestart!")
    print("📊 Dashboard beschikbaar op: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)