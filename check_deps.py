import sys
import importlib.util
import subprocess

def check_package(package_name, import_name=None):
    """Проверяет, установлен ли пакет"""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    return spec is not None

def main():
    # Основные зависимости
    core_packages = {
        "streamlit": "streamlit",
        "qdrant-client": "qdrant_client",
        "sentence-transformers": "sentence_transformers",
        "torch": "torch",
        "PyPDF2": "PyPDF2",
        "python-docx": "docx",
        "numpy": "numpy",
        "requests": "requests",
        "aiohttp": "aiohttp",
        "asyncio-mqtt": "asyncio_mqtt"
    }
    
    # Опциональные зависимости
    optional_packages = {
        "pdfplumber": "pdfplumber",
        "python-pptx": "pptx",
        "openpyxl": "openpyxl"
    }
    
    print("🔍 Проверка установленных библиотек...\n")
    
    # Проверка основных зависимостей
    core_missing = []
    core_installed = []
    
    print("📦 Основные зависимости:")
    for package, import_name in core_packages.items():
        if check_package(package, import_name):
            core_installed.append(package)
            print(f"   ✅ {package}")
        else:
            core_missing.append(package)
            print(f"   ❌ {package}")
    
    # Проверка опциональных зависимостей
    optional_missing = []
    optional_installed = []
    
    print("\n📎 Опциональные зависимости:")
    for package, import_name in optional_packages.items():
        if check_package(package, import_name):
            optional_installed.append(package)
            print(f"   ✅ {package}")
        else:
            optional_missing.append(package)
            print(f"   ⚠️  {package} (необязательно)")
    
    # Итоговый отчет
    print(f"\n📊 Результат:")
    print(f"   Основные установлено: {len(core_installed)}/{len(core_packages)}")
    print(f"   Основные отсутствуют: {len(core_missing)}")
    print(f"   Опциональные установлено: {len(optional_installed)}/{len(optional_packages)}")
    print(f"   Опциональные отсутствуют: {len(optional_missing)}")
    
    # Проверка критических отсутствующих пакетов
    critical_missing = []
    if "torch" in core_missing:
        critical_missing.append("torch")
    if "sentence-transformers" in core_missing:
        critical_missing.append("sentence-transformers")
    if "qdrant-client" in core_missing:
        critical_missing.append("qdrant-client")
    
    if core_missing:
        print(f"\n🚨 Критические отсутствующие зависимости:")
        for pkg in critical_missing:
            print(f"   - {pkg}")
        
        print(f"\n📦 Все отсутствующие основные зависимости:")
        for pkg in core_missing:
            print(f"   - {pkg}")
        
        if optional_missing:
            print(f"\n📎 Отсутствующие опциональные зависимости:")
            for pkg in optional_missing:
                print(f"   - {pkg}")
        
        return 1
    else:
        print(f"\n🎉 Все основные зависимости установлены!")
        if optional_missing:
            print(f"ℹ️  Некоторые опциональные зависимости отсутствуют (это нормально)")
        return 0

if __name__ == "__main__":
    sys.exit(main())