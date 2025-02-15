# Reclaim

### Setting up the front-end

Requirements:
- Node v23.0 or greater
  
Clone the repo
```
git clone https://github.com/scottallencampbell/Reclaim.Web
```
Set two environment variables to the location of the backend
```
SET REACT_APP_API_URL=http://localhost:50000
SET REACT_APP_SIGNALR_URL=http://localhost:50000
```
Launch the website
```
npm run start
```

<br>

### Setting up the back-end
  
Requirements:
- Visual Studio 2022
- SQL Server 2022, Developer or greater
    
Clone the repo 
```
git clone https://github.com/scottallencampbell/Reclaim.Api
```
Restore the Reclaim.bak database backup
Set an environment variable to the SQL Server connection string for the local database
```
SET RECLAIM_API_CONNECTION_STRING=
Server=.;Database=reclaim;User Id=reclaim-api;Password=reclaim-api;TrustServerCertificate=True
```
Create a SQL Server user as required  

Adjust values as needed in the ApplicationSetting table

Launch the webapi using Visual Studio
