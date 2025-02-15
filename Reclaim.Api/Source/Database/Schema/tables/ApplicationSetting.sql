CREATE TABLE [dbo].[ApplicationSetting] (
   [ApplicationSettingID] [int] NOT NULL,
   [Name] [nvarchar](50) NOT NULL,
   [Value] [nvarchar](max) NOT NULL,
   [IsSecret] [bit] NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_ApplicationSetting] PRIMARY KEY CLUSTERED ([ApplicationSettingID])
)

CREATE UNIQUE NONCLUSTERED INDEX [IX_ApplicationSetting] ON [dbo].[ApplicationSetting] ([Name])

GO
