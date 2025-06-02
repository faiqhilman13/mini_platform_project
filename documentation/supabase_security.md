# 🔐 Supabase Security Guide for Mini.ML Platform

**🎯 Mission Critical:** Secure your ML platform from Day 1. This guide covers hardening your Supabase setup against common attack vectors.

---

## 🚨 **IMMEDIATE SECURITY ACTIONS REQUIRED**

### 1. **Enable RLS on ALL Tables** (CRITICAL)
```sql
-- Apply to ALL 6 tables immediately
ALTER TABLE uploadedfilelog ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipelinerun ENABLE ROW LEVEL SECURITY;
ALTER TABLE dataprofiling ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_pipeline_run ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_experiment ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_model ENABLE ROW LEVEL SECURITY;
```

### 2. **Create Basic RLS Policies**
```sql
-- Users can only access their own files
CREATE POLICY "Users can manage their own files" ON uploadedfilelog
    FOR ALL USING (created_by = auth.uid());

-- Users can only access their own pipeline runs
CREATE POLICY "Users can manage their own pipeline runs" ON pipelinerun
    FOR ALL USING (created_by = auth.uid());

-- Users can only access profiling data for their files
CREATE POLICY "Users can access their file profiling" ON dataprofiling
    FOR ALL USING (
        file_id IN (
            SELECT id FROM uploadedfilelog WHERE created_by = auth.uid()
        )
    );

-- Users can only access their ML pipeline runs
CREATE POLICY "Users can manage their ML runs" ON ml_pipeline_run
    FOR ALL USING (created_by = auth.uid());

-- Users can only access their experiments
CREATE POLICY "Users can manage their experiments" ON ml_experiment
    FOR ALL USING (created_by = auth.uid());

-- Users can only access models from their pipeline runs
CREATE POLICY "Users can access their models" ON ml_model
    FOR ALL USING (
        pipeline_run_id IN (
            SELECT id FROM ml_pipeline_run WHERE created_by = auth.uid()
        )
    );
```

---

## 🧱 **AUTHENTICATION & AUTHORIZATION**

### **Auth Configuration** (Dashboard: Authentication > Settings)
- [ ] **Disable self-registration** unless you want public signups
- [ ] **Enable email confirmation** for all signups
- [ ] **Set strong password requirements** (min 8 chars, complexity)
- [ ] **Configure allowed origins** (remove wildcards in production)
- [ ] **Enable MFA** for admin accounts

### **OAuth Providers** (Only enable what you need)
- [ ] **Google:** Enable only if required
- [ ] **GitHub:** Enable only if required  
- [ ] **Disable unused providers** (reduces attack surface)

### **Auth Hooks** (Advanced)
```sql
-- Example: Auto-assign user roles on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    -- Validate email domain for restricted access
    IF NEW.email NOT LIKE '%@yourcompany.com' THEN
        RAISE EXCEPTION 'Unauthorized domain';
    END IF;
    
    -- Insert user profile with default role
    INSERT INTO public.user_profiles (id, email, role)
    VALUES (NEW.id, NEW.email, 'user');
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger on auth.users insert
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
```

---

## 🗃️ **DATABASE SECURITY**

### **Connection Security**
- [ ] **Use connection pooling** (PgBouncer) in production
- [ ] **Enable SSL** for all connections
- [ ] **Restrict IP access** in production (Database > Settings)
- [ ] **Regular password rotation** for service accounts

### **Advanced RLS Patterns for ML Platform**

#### **Time-based Access Control**
```sql
-- Expire old profiling data access
CREATE POLICY "Access valid profiling data" ON dataprofiling
    FOR SELECT USING (
        created_by = auth.uid() AND 
        expires_at > NOW()
    );
```

#### **Hierarchical Access Control**
```sql
-- Admins can see all data (be very careful!)
CREATE POLICY "Admin access" ON uploadedfilelog
    FOR ALL USING (
        auth.uid() IN (
            SELECT id FROM auth.users 
            WHERE raw_user_meta_data->>'role' = 'admin'
        )
    );
```

#### **Resource Limits**
```sql
-- Limit number of files per user
CREATE POLICY "File upload limits" ON uploadedfilelog
    FOR INSERT WITH CHECK (
        (SELECT COUNT(*) FROM uploadedfilelog WHERE created_by = auth.uid()) < 100
    );
```

---

## 📁 **STORAGE SECURITY**

### **Bucket Configuration**
```sql
-- Create private bucket for ML models
INSERT INTO storage.buckets (id, name, public)
VALUES ('ml-models', 'ml-models', false);

-- Create private bucket for datasets
INSERT INTO storage.buckets (id, name, public)
VALUES ('datasets', 'datasets', false);
```

### **Storage RLS Policies**
```sql
-- Users can only upload to their own folder
CREATE POLICY "User folder upload" ON storage.objects
    FOR INSERT WITH CHECK (
        bucket_id = 'datasets' AND 
        auth.uid()::text = (storage.foldername(name))[1]
    );

-- Users can only access their own files
CREATE POLICY "User file access" ON storage.objects
    FOR SELECT USING (
        bucket_id = 'datasets' AND 
        auth.uid()::text = (storage.foldername(name))[1]
    );
```

### **File Upload Validation**
```sql
-- Validate file types and sizes
CREATE OR REPLACE FUNCTION validate_file_upload()
RETURNS TRIGGER AS $$
BEGIN
    -- Check file size (max 100MB)
    IF NEW.size_bytes > 104857600 THEN
        RAISE EXCEPTION 'File too large';
    END IF;
    
    -- Check allowed content types for ML platform
    IF NEW.content_type NOT IN (
        'text/csv',
        'application/json',
        'application/pdf',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    ) THEN
        RAISE EXCEPTION 'Invalid file type';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER validate_file_upload_trigger
    BEFORE INSERT ON uploadedfilelog
    FOR EACH ROW EXECUTE FUNCTION validate_file_upload();
```

---

## 🔍 **API SECURITY**

### **Environment Variables**
```bash
# .env.local (never commit!)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key  # Safe for frontend
SUPABASE_SERVICE_KEY=your-service-key  # SERVER ONLY!

# Production environment
SUPABASE_URL=https://your-prod-project.supabase.co
```

### **Client-Side Security**
```typescript
// ❌ NEVER do this
const supabase = createClient(url, serviceKey); // Admin access!

// ✅ Always use anon key on frontend
const supabase = createClient(url, anonKey);

// ✅ Server-side admin operations only
// app/api/admin/route.ts
const supabaseAdmin = createClient(url, serviceKey);
```

### **Rate Limiting** (Edge Functions)
```typescript
// supabase/functions/upload/index.ts
const rateLimiter = new Map();

Deno.serve(async (req) => {
    const userId = await getUserId(req);
    const now = Date.now();
    const windowMs = 60000; // 1 minute
    const maxRequests = 10;
    
    const userRequests = rateLimiter.get(userId) || [];
    const recentRequests = userRequests.filter(time => now - time < windowMs);
    
    if (recentRequests.length >= maxRequests) {
        return new Response('Rate limit exceeded', { status: 429 });
    }
    
    recentRequests.push(now);
    rateLimiter.set(userId, recentRequests);
    
    // Continue with upload logic...
});
```

---

## 🛡️ **PRODUCTION HARDENING**

### **Database Configuration**
- [ ] **Enable audit logging** (Database > Logs)
- [ ] **Set up monitoring** for suspicious activity
- [ ] **Regular backups** with encryption
- [ ] **Database extensions** only enable what you need

### **Network Security**
```sql
-- Restrict database access to specific IPs
-- (Configure in Dashboard: Settings > Database)

-- Monitor failed login attempts
SELECT created_at, raw_user_meta_data->>'ip' as ip, email
FROM auth.audit_log_entries 
WHERE event_type = 'SIGN_IN_FAILURE'
ORDER BY created_at DESC;
```

### **Content Security Policy**
```html
<!-- Add to your HTML head -->
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; 
               connect-src 'self' https://*.supabase.co;
               script-src 'self' 'unsafe-inline';
               style-src 'self' 'unsafe-inline';">
```

---

## 🧪 **SECURITY TESTING**

### **Manual Testing Checklist**
- [ ] Try accessing other users' data with different auth tokens
- [ ] Test file upload with malicious file types
- [ ] Attempt SQL injection in text fields
- [ ] Test RLS bypass with direct API calls
- [ ] Verify storage bucket access controls

### **Automated Testing**
```typescript
// tests/security.test.ts
describe('Security Tests', () => {
    test('Cannot access other users files', async () => {
        const { data: user1Files } = await supabase1
            .from('uploadedfilelog')
            .select('*');
            
        const { data: user2Files } = await supabase2
            .from('uploadedfilelog')
            .select('*');
            
        // user2 should not see user1's files
        expect(user2Files).not.toContain(user1Files[0]);
    });
});
```

### **PostgREST Direct Testing**
```bash
# Test RLS directly
curl -X GET "https://your-project.supabase.co/rest/v1/uploadedfilelog" \
  -H "apikey: YOUR_ANON_KEY" \
  -H "Authorization: Bearer ANOTHER_USER_JWT"
```

---

## 🚨 **MONITORING & INCIDENT RESPONSE**

### **Set Up Alerts**
```sql
-- Monitor suspicious activity
CREATE OR REPLACE FUNCTION notify_suspicious_activity()
RETURNS TRIGGER AS $$
BEGIN
    -- Alert on bulk data access
    IF (SELECT COUNT(*) FROM uploadedfilelog WHERE created_by = NEW.created_by) > 50 THEN
        -- Send alert (integrate with your monitoring system)
        PERFORM pg_notify('security_alert', 
            'Bulk file access by user: ' || NEW.created_by::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

### **Log Analysis**
- [ ] **Monitor auth.audit_log_entries** for failed logins
- [ ] **Track API usage patterns** in Supabase Dashboard
- [ ] **Set up Slack/email alerts** for security events
- [ ] **Regular access reviews** of user permissions

---

## 📚 **QUICK REFERENCE**

### **Emergency Commands**
```sql
-- Disable a compromised user
UPDATE auth.users SET email_confirmed_at = NULL WHERE id = 'user-id';

-- Revoke all sessions for a user
DELETE FROM auth.sessions WHERE user_id = 'user-id';

-- Temporarily disable RLS (emergency only!)
ALTER TABLE table_name DISABLE ROW LEVEL SECURITY;
```

### **Security Health Check Script**
```sql
-- Run this monthly
SELECT 
    schemaname,
    tablename,
    rowsecurity as rls_enabled
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY tablename;
```

---

## ⚡ **ACTION ITEMS FOR MINI.ML**

### **Phase 1: Immediate (This Week)**
- [ ] Enable RLS on all 6 tables
- [ ] Create basic user-scoped policies
- [ ] Configure Auth settings
- [ ] Set up private storage buckets

### **Phase 2: Security Hardening (Next Sprint)**
- [ ] Implement file upload validation
- [ ] Add rate limiting to critical endpoints
- [ ] Set up monitoring and alerting
- [ ] Create security testing suite

### **Phase 3: Advanced Security (Month 2)**
- [ ] Implement advanced RLS patterns
- [ ] Add audit logging
- [ ] Set up IP restrictions for production
- [ ] Create incident response procedures

---

**🔥 Remember:** Security is not a feature you add later—it's the foundation you build on. Every ML model trained on compromised data is a liability.

**Next Step:** Apply the immediate actions from Phase 1 to your current Supabase setup before proceeding with any production deployments. 