ALTER TABLE [dbo].[EmailTemplate] ADD CONSTRAINT [DF_EmailTemplate_Subject] DEFAULT (N'd') FOR [Subject]
GO
ALTER TABLE [dbo].[EmailTemplate] ADD CONSTRAINT [DF_EmailTemplate_Preheader] DEFAULT (N'x') FOR [Preheader]
GO
ALTER TABLE [dbo].[EmailTemplate] ADD CONSTRAINT [DF_EmailTemplate_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[EmailTemplate] ADD CONSTRAINT [DF_EmailTemplate_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
