# Langfuse Setup Guide

## Launch Langfuse services
Start up the docker compose stack: `docker-compose up -d`.

Langfuse currently does not offer an Admin API, as discussed in this [GitHub discussion](https://github.com/orgs/langfuse/discussions/1007). Therefore, setting up the admin user, creating an organization, and establishing a Langfuse project must be done manually.

Access Langfuse at: [http://localhost:3000/](http://localhost:3000/)

## Setup Steps

1. **Admin User Creation**
   - The first user to sign up will be the admin.
   - Navigate to the login screen and click on "Sign Up" or use this direct link: [http://localhost:3000/auth/sign-up](http://localhost:3000/auth/sign-up).

2. **Organization Setup**
   - Click on `Create new organization`.
   - Enter the organization name as `local` and click `Next`.
   - Skip the invite members step by clicking `Next`.

3. **Project Creation**
   - Enter the project name as `eval-tests` and click `Create`.

4. **API Key Generation**
   - Go to the project settings and click on `API Keys`.
   - Select `Create new API Keys`.
   - Copy the generated credentials for use in the application you are observing.

Follow these steps to successfully set up your Langfuse environment.