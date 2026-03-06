# Quick Setup Guide

## Step 1: Create Supabase Project

1. Go to https://supabase.com and sign up or log in
2. Click "New Project"
3. Fill in the project details:
   - Project name: "Dementia Detection System"
   - Database password: (choose a strong password)
   - Region: (choose closest to you)
4. Click "Create new project" and wait for setup to complete

## Step 2: Get Your API Credentials

1. In your Supabase project dashboard, click the "Settings" icon
2. Navigate to "API" in the settings menu
3. Copy the following values:
   - **Project URL** (under "Project API keys")
   - **anon public** key (under "Project API keys")

## Step 3: Configure Environment Variables

1. In your project folder, create a `.env` file:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and paste your credentials:
   ```
   VITE_SUPABASE_URL=https://your-project-id.supabase.co
   VITE_SUPABASE_ANON_KEY=your-anon-key-here
   ```

## Step 4: Verify Database Schema

The database schema should already be applied. To verify:

1. In Supabase dashboard, go to "Table Editor"
2. You should see three tables:
   - `patients`
   - `assessments`
   - `module_results`

If tables are missing, they were already created via migration.

## Step 5: Install Dependencies and Run

```bash
npm install
npm run dev
```

## Step 6: Create Your Neurologist Account

1. Open http://localhost:5173 in your browser
2. You'll see the login page
3. Use your email and create a password to sign up
4. You'll be automatically logged in

## Troubleshooting

### "Missing Supabase environment variables" error
- Make sure your `.env` file exists and has the correct values
- Restart the dev server after creating/editing `.env`

### "Authentication failed" error
- Check that your Supabase project URL and anon key are correct
- Verify your Supabase project is active and running

### Database connection issues
- Ensure your Supabase project is not paused (free tier projects pause after inactivity)
- Check your internet connection
- Verify the database migrations were applied successfully

### Build warnings about chunk size
- This is normal for development
- The warnings can be safely ignored
- For production, consider code splitting (optional optimization)

## Testing the System

1. **Create a Test Patient**
   - Click "New Patient"
   - Fill in test data
   - Save the patient

2. **Run an Assessment**
   - Click on the patient card
   - Start the first module (Voice Analysis)
   - Watch the progress animation
   - Each module will automatically proceed to the next
   - View results after all modules complete

3. **Download PDF Report**
   - From the results page, click "Download PDF Report"
   - A comprehensive PDF will be generated and downloaded

## Production Deployment

For deploying to production:

1. Build the project:
   ```bash
   npm run build
   ```

2. Deploy the `dist` folder to your hosting service (Vercel, Netlify, etc.)

3. Add environment variables to your hosting platform:
   - `VITE_SUPABASE_URL`
   - `VITE_SUPABASE_ANON_KEY`

4. Enable RLS policies in Supabase (already configured)

5. Consider disabling Supabase email confirmation in production for easier testing, or configure an email provider

## Next Steps

- Integrate real Python analysis scripts via Supabase Edge Functions
- Add more comprehensive validation
- Implement historical data tracking
- Add data export features
- Create admin panel for user management
- Add email notifications for completed assessments

## Support

For technical support or questions:
- Check Supabase documentation: https://supabase.com/docs
- Review React documentation: https://react.dev
- Check the project README.md for detailed information
