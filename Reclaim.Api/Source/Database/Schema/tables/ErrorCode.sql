CREATE TABLE [dbo].[ErrorCode] (
   [ErrorCodeID] [int] NOT NULL,
   [Code] [nvarchar](50) NOT NULL,
   [Description] [nvarchar](500) NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_ErrorCode] PRIMARY KEY CLUSTERED ([ErrorCodeID])
)

CREATE UNIQUE NONCLUSTERED INDEX [IX_ErrorCode] ON [dbo].[ErrorCode] ([Code])

GO
