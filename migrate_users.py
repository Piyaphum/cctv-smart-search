import yaml
import config
from supabase import create_client

def migrate():
    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    
    try:
        with open('auth_config.yaml', 'r', encoding='utf-8') as f:
            auth_config = yaml.safe_load(f)
            
        users = auth_config.get('credentials', {}).get('usernames', {})
        
        for uname, details in users.items():
            data = {
                "username": uname,
                "name": details.get("name", "Unknown"),
                "email": details.get("email", "unknown@example.com"),
                "password_hash": details.get("password", ""),
                "role": details.get("role", "viewer")
            }
            # Upsert user to avoid duplicates if ran multiple times
            supabase.table('users').upsert(data).execute()
        
        print(f"Successfully migrated {len(users)} users to Supabase cloud.")
    except Exception as e:
        print(f"Migration failed: {e}")

if __name__ == "__main__":
    migrate()
