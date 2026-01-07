"""
Test WRDS connection and setup

This script tests if WRDS is properly configured and provides setup instructions if needed.
"""

import os

print("="*80)
print("WRDS CONNECTION TEST")
print("="*80)

# Test 1: Check if wrds module is installed
print("\n[1/3] Checking if wrds module is installed...")
try:
    import wrds
    print("  ✓ wrds module found")
except ImportError:
    print("  ✗ wrds module not found")
    print("\n  Install with: pip install wrds")
    exit(1)

# Test 2: Check for .pgpass file
print("\n[2/3] Checking for .pgpass file...")
pgpass_path = os.path.expanduser("~/.pgpass")
if os.path.exists(pgpass_path):
    print(f"  ✓ .pgpass file found at {pgpass_path}")
    with open(pgpass_path, 'r') as f:
        lines = f.readlines()
        wrds_lines = [l for l in lines if 'wrds' in l.lower()]
        if wrds_lines:
            print(f"  ✓ WRDS credentials found in .pgpass")
        else:
            print(f"  ✗ No WRDS credentials in .pgpass")
            print("\n  Add a line like: wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_USERNAME:YOUR_PASSWORD")
else:
    print(f"  ✗ .pgpass file not found")
    print("\n  Create ~/.pgpass with:")
    print("    wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_USERNAME:YOUR_PASSWORD")
    print("  Then set permissions: chmod 600 ~/.pgpass")

# Test 3: Try to connect
print("\n[3/3] Testing WRDS connection...")
try:
    db = wrds.Connection(wrds_username='ofs4963')
    print("  ✓ Successfully connected to WRDS")

    # Test a simple query
    print("\n  Testing query...")
    result = db.raw_sql("SELECT COUNT(*) as cnt FROM crsp.dsf WHERE date = '2020-01-02'")
    count = result.iloc[0]['cnt']
    print(f"  ✓ Query successful: {count:,} observations on 2020-01-02")

    db.close()
    print("\n  ✓ Connection closed")

    print("\n" + "="*80)
    print("SUCCESS: WRDS is properly configured")
    print("="*80)
    print("\nYou can now run: python 04_wrds_data_merge.py --test")

except Exception as e:
    print(f"  ✗ Connection failed: {e}")
    print("\n" + "="*80)
    print("SETUP INSTRUCTIONS")
    print("="*80)
    print("""
1. Create a .pgpass file in your home directory:
   Location: ~/.pgpass (Linux/Mac) or %APPDATA%\postgresql\pgpass.conf (Windows)

2. Add this line (replace YOUR_PASSWORD):
   wrds-pgdata.wharton.upenn.edu:9737:wrds:ofs4963:YOUR_PASSWORD

3. Set file permissions:
   Linux/Mac: chmod 600 ~/.pgpass
   Windows: No chmod needed

4. Alternatively, set environment variable:
   export PGPASSWORD=your_password

5. Test connection:
   python test_wrds_connection.py
""")
