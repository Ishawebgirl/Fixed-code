CREATE TABLE [dbo].[JobStatus] (
   [JobStatusID] [int] NOT NULL,
   [Code] [nvarchar](50) NOT NULL,
   [Description] [nvarchar](500) NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_JobStatus] PRIMARY KEY CLUSTERED ([JobStatusID])
)

CREATE UNIQUE NONCLUSTERED INDEX [IX_JobStatus] ON [dbo].[JobStatus] ([Code])

GO
