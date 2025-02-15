CREATE TABLE [dbo].[EmailTemplate] (
   [EmailTemplateID] [int] NOT NULL,
   [Code] [nvarchar](50) NOT NULL,
   [ExternalID] [nvarchar](100) NOT NULL,
   [Subject] [nvarchar](100) NOT NULL,
   [Preheader] [nvarchar](100) NOT NULL,
   [Body] [nvarchar](max) NOT NULL,
   [HighlightColor] [nvarchar](10) NOT NULL,
   [ActionButtonText] [nvarchar](50) NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_EmailTemplate] PRIMARY KEY CLUSTERED ([EmailTemplateID])
)

CREATE UNIQUE NONCLUSTERED INDEX [IX_EmailTemplate] ON [dbo].[EmailTemplate] ([Code])

GO
