# Lean SaaS Runway - Mini IDP Platform

**Philosophy: Build Less, Integrate More, Scale Faster**

This document outlines a lean approach to transforming Mini IDP into a production SaaS by leveraging external services and keeping the codebase minimal. Focus on core differentiators, outsource everything else.

## 🎯 **Phase 1: Authentication & Payments (Lean Foundation)**
*Timeline: Week 1-2 | Priority: Critical*

### 1.1 Authentication (Use Supabase Auth)
- [ ] **P1.1.1: Replace custom auth with Supabase**
  - [ ] Set up Supabase project (free tier)
  - [ ] Add `@supabase/supabase-js` to frontend
  - [ ] Replace FastAPI auth with Supabase JWT verification
  - [ ] Add organization_id to Supabase user metadata
  - [ ] **Result**: Full auth system in ~50 lines of code

- [ ] **P1.1.2: Multi-tenant data isolation**
  - [ ] Add RLS (Row Level Security) policies in Supabase
  - [ ] Update database queries to filter by organization_id
  - [ ] **Result**: Tenant isolation without custom middleware

### 1.2 Payments (Use Stripe + n8n)
- [ ] **P1.2.1: Stripe setup (minimal)**
  - [ ] Create Stripe account and get API keys
  - [ ] Add 3 products: Free ($0), Pro ($29), Enterprise ($199)
  - [ ] Use Stripe Customer Portal for all billing UI
  - [ ] **Result**: Zero billing code to maintain

- [ ] **P1.2.2: n8n automation for billing**
  - [ ] Set up n8n cloud instance
  - [ ] Create workflow: Stripe webhook → Update user limits in database
  - [ ] Create workflow: Usage threshold → Send email via SendGrid
  - [ ] **Result**: Billing automation without custom code

---

## 💳 **Phase 2: Usage Limits & Core Features**
*Timeline: Week 3-4 | Priority: High*

### 2.1 Usage Tracking (Minimal Code)
- [ ] **P2.1.1: Simple usage middleware**
  - [ ] Add single middleware function to track API calls
  - [ ] Store usage in simple table: `user_id, date, api_calls, storage_mb`
  - [ ] Check limits before expensive operations only
  - [ ] **Result**: ~30 lines of usage tracking code

- [ ] **P2.1.2: n8n usage automation**
  - [ ] Daily workflow: Aggregate usage → Update user dashboard
  - [ ] Alert workflow: 80% limit reached → Email user
  - [ ] Reset workflow: Monthly usage reset
  - [ ] **Result**: Usage management without cron jobs

### 2.2 Feature Flags (Use PostHog)
- [ ] **P2.2.1: PostHog integration**
  - [ ] Add PostHog SDK to frontend and backend
  - [ ] Create feature flags for: advanced_ml, api_access, white_label
  - [ ] Gate features based on subscription tier
  - [ ] **Result**: Feature management without custom code

---

## 🏢 **Phase 3: Enterprise Essentials (Service-First)**
*Timeline: Week 5-8 | Priority: Medium*

### 3.1 Customer Support (Use Intercom)
- [ ] **P3.1.1: Intercom setup**
  - [ ] Add Intercom widget to frontend
  - [ ] Set up automated onboarding messages
  - [ ] Create help articles for common issues
  - [ ] **Result**: Full support system, zero code

### 3.2 Analytics (Use PostHog + Mixpanel)
- [ ] **P3.2.1: Event tracking**
  - [ ] Track key events: signup, upload, train_model, api_call
  - [ ] Set up conversion funnels in PostHog
  - [ ] Create revenue dashboard in Mixpanel
  - [ ] **Result**: Full analytics without custom dashboards

### 3.3 Email & Notifications (Use n8n + SendGrid)
- [ ] **P3.3.1: n8n email workflows**
  - [ ] Welcome email sequence for new users
  - [ ] Usage limit warnings
  - [ ] Model training completion notifications
  - [ ] Monthly usage reports
  - [ ] **Result**: Email automation without custom email service

### 3.4 API Documentation (Use Mintlify)
- [ ] **P3.4.1: Beautiful docs**
  - [ ] Set up Mintlify for API documentation
  - [ ] Auto-generate from OpenAPI spec
  - [ ] Add code examples and SDKs
  - [ ] **Result**: Professional docs without maintenance

---

## 🤖 **Phase 4: Advanced Features (Smart Outsourcing)**
*Timeline: Week 9-12 | Priority: Medium*

### 4.1 Model Serving (Use Replicate)
- [ ] **P4.1.1: Replicate integration**
  - [ ] Push trained models to Replicate
  - [ ] Create prediction endpoints via Replicate API
  - [ ] Add model versioning through Replicate
  - [ ] **Result**: Production ML serving without infrastructure

### 4.2 File Storage (Use Supabase Storage)
- [ ] **P4.2.1: Replace local storage**
  - [ ] Migrate to Supabase Storage buckets
  - [ ] Add organization-based bucket policies
  - [ ] Implement CDN for file serving
  - [ ] **Result**: Scalable storage without S3 complexity

### 4.3 Database (Use Supabase PostgreSQL)
- [ ] **P4.3.1: Migrate from SQLite**
  - [ ] Export data from SQLite
  - [ ] Set up Supabase PostgreSQL
  - [ ] Update connection strings
  - [ ] **Result**: Production database without DevOps

### 4.4 Monitoring (Use Better Stack)
- [ ] **P4.4.1: Uptime monitoring**
  - [ ] Set up Better Stack for uptime monitoring
  - [ ] Add error tracking with Sentry
  - [ ] Create status page
  - [ ] **Result**: Full monitoring without custom dashboards

---

## 🚀 **Phase 5: Scale & Growth (Service-Heavy)**
*Timeline: Week 13-16 | Priority: Low*

### 5.1 Customer Success (Use Pendo)
- [ ] **P5.1.1: User onboarding**
  - [ ] Add Pendo for in-app guidance
  - [ ] Create product tours for new features
  - [ ] Track feature adoption
  - [ ] **Result**: Customer success without custom code

### 5.2 A/B Testing (Use PostHog)
- [ ] **P5.2.1: Experiment framework**
  - [ ] Set up A/B tests for pricing page
  - [ ] Test different onboarding flows
  - [ ] Optimize conversion funnels
  - [ ] **Result**: Experimentation without custom framework

### 5.3 SEO & Content (Use Webflow)
- [ ] **P5.3.1: Marketing site**
  - [ ] Build marketing site in Webflow
  - [ ] Add blog and documentation
  - [ ] Implement SEO best practices
  - [ ] **Result**: Marketing presence without frontend maintenance

---

## 🛠️ **Lean Tech Stack**

### Core Application (Minimal Code)
```
Backend: FastAPI (current) - Keep lean, only business logic
Frontend: React (current) - Focus on core features only
Database: Supabase PostgreSQL - Managed, with RLS
Auth: Supabase Auth - Zero auth code
Storage: Supabase Storage - No S3 complexity
```

### External Services (Zero Code)
```
Payments: Stripe + Customer Portal
Automation: n8n Cloud
Analytics: PostHog + Mixpanel  
Support: Intercom
Email: SendGrid (via n8n)
Monitoring: Better Stack + Sentry
Docs: Mintlify
ML Serving: Replicate
Feature Flags: PostHog
A/B Testing: PostHog
Customer Success: Pendo
```

---

## 💰 **Lean Pricing Strategy**

### Service Costs (Monthly)
```
Supabase Pro: $25/month (auth + db + storage)
n8n Cloud: $20/month (automation)
PostHog: $0-50/month (analytics + flags)
Stripe: 2.9% + 30¢ per transaction
Intercom: $39/month (support)
SendGrid: $15/month (email)
Better Stack: $18/month (monitoring)
Mintlify: $120/month (docs)
Total: ~$250/month base cost
```

### Revenue Targets
```
Month 1: $1,000 MRR (break-even at ~35 customers)
Month 3: $5,000 MRR (profitable)
Month 6: $15,000 MRR (sustainable growth)
Month 12: $50,000 MRR (scale mode)
```

---

## 📋 **Implementation Priority**

### Week 1-2: Foundation
- [ ] Set up Supabase project
- [ ] Migrate auth to Supabase
- [ ] Set up Stripe products
- [ ] Create n8n billing automation

### Week 3-4: Core Features  
- [ ] Add usage tracking middleware
- [ ] Set up PostHog feature flags
- [ ] Create pricing page
- [ ] Launch beta with 10 users

### Week 5-8: Growth Tools
- [ ] Add Intercom support
- [ ] Set up email automation
- [ ] Create analytics dashboards
- [ ] Launch public beta

### Week 9-12: Scale Prep
- [ ] Migrate to production database
- [ ] Add monitoring and alerts
- [ ] Set up model serving
- [ ] Optimize for 100+ users

---

## 🎯 **Lean Development Rules**

### Code Philosophy
1. **Don't build what you can buy** - Use services for non-core features
2. **Don't build what you can integrate** - APIs over custom solutions
3. **Don't optimize prematurely** - Scale when you have the problem
4. **Don't maintain what others maintain better** - Managed services first

### Service Selection Criteria
1. **Has generous free tier** - Minimize upfront costs
2. **Scales with usage** - Pay as you grow
3. **Good API/integration** - Easy to connect
4. **Reliable vendor** - Won't disappear overnight

### When to Build vs Buy
**Build**: Core ML algorithms, unique business logic, competitive differentiators
**Buy**: Auth, payments, email, monitoring, analytics, support, docs

---

## 🚀 **Next Actions (This Week)**

### Day 1-2: Setup External Services
1. Create Supabase account and project
2. Set up Stripe account with 3 products
3. Sign up for n8n cloud
4. Create PostHog account

### Day 3-5: Core Integration
1. Replace FastAPI auth with Supabase JWT
2. Add organization_id to user metadata
3. Create Stripe webhook → n8n → database workflow
4. Add basic usage tracking middleware

### Day 6-7: Launch Prep
1. Set up pricing page with Stripe checkout
2. Add PostHog analytics tracking
3. Create simple onboarding flow
4. Deploy to production

**Goal**: Launch paid beta by end of Week 2 with minimal code changes.

This lean approach gets you to market faster, reduces maintenance burden, and lets you focus on what makes your platform unique - the ML capabilities.

---

## 🔧 **n8n Workflow Examples**

### Billing Automation Workflow
```
Trigger: Stripe Webhook (subscription.updated)
→ Extract customer_id and subscription_status
→ Query Supabase for user by stripe_customer_id
→ Update user subscription_tier in database
→ If downgrade: Send email with retention offer
→ If upgrade: Send welcome email with new features
```

### Usage Monitoring Workflow
```
Trigger: Daily at 9 AM
→ Query database for yesterday's usage by user
→ Calculate percentage of monthly limit used
→ If >80%: Send warning email via SendGrid
→ If >100%: Disable API access and send upgrade email
→ Log all actions to monitoring dashboard
```

### Customer Onboarding Workflow
```
Trigger: New user signup (PostHog event)
→ Wait 1 hour
→ Send welcome email with getting started guide
→ Wait 3 days
→ Check if user uploaded first file
→ If no: Send tutorial email
→ If yes: Send advanced features email
→ Schedule follow-up in 1 week
```

This approach lets you build complex automation without writing backend code! 