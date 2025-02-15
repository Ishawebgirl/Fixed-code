CREATE TABLE [dbo].[Job] (
   [JobID] [int] NOT NULL,
   [JobTypeID] [int] NOT NULL,
   [JobStatusID] [int] NOT NULL,
   [Name] [nvarchar](50) NOT NULL,
   [Description] [nvarchar](500) NOT NULL,
   [Interval] [int] NOT NULL,
   [Timeout] [int] NOT NULL,
   [NextEvent] [datetime] NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_Job] PRIMARY KEY CLUSTERED ([JobID])
)


GO
