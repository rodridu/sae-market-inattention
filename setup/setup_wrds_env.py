"""
Setup WRDS environment variables

Run this once to configure WRDS credentials, then run the data merge script.
"""

import os
import getpass

print("="*80)
print("WRDS ENVIRONMENT SETUP")
print("="*80)

# Get credentials
username = input("\nEnter your WRDS username [ofs4963]: ").strip() or "ofs4963"
password = getpass.getpass("Enter your WRDS password: ")

# Test connection
print("\nTesting WRDS connection...")
try:
    import wrds
    db = wrds.Connection(wrds_username=username)
    print("✓ Connected successfully!")

    # Simple test query
    result = db.raw_sql("SELECT 1 as test")
    print("✓ Query test passed!")

    db.close()

    # Save to environment file
    env_file = os.path.join(os.path.dirname(__file__), ".env_wrds")
    with open(env_file, 'w') as f:
        f.write(f"WRDS_USERNAME={username}\n")

    print(f"\n✓ Configuration saved to {env_file}")
    print("\nNow you can run: python 04_wrds_data_merge.py --test")

except Exception as e:
    print(f"\n✗ Connection failed: {e}")
    print("\nPlease check your credentials and try again.")
