CREATE TABLE [dbo].[JobType] (
   [JobTypeID] [int] NOT NULL,
   [Code] [nvarchar](50) NOT NULL,
   [Description] [nvarchar](500) NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_JobType] PRIMARY KEY CLUSTERED ([JobTypeID])
)

CREATE UNIQUE NONCLUSTERED INDEX [IX_JobType] ON [dbo].[JobType] ([Code])

GO
