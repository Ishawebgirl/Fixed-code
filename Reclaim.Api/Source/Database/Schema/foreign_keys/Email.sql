ALTER TABLE [dbo].[Email] WITH CHECK ADD CONSTRAINT [FK_Email_Account]
   FOREIGN KEY([AccountID]) REFERENCES [dbo].[Account] ([AccountID])

GO
ALTER TABLE [dbo].[Email] WITH CHECK ADD CONSTRAINT [FK_Email_EmailStatus]
   FOREIGN KEY([EmailStatusID]) REFERENCES [dbo].[EmailStatus] ([EmailStatusID])

GO
ALTER TABLE [dbo].[Email] WITH CHECK ADD CONSTRAINT [FK_Email_EmailTemplate]
   FOREIGN KEY([EmailTemplateID]) REFERENCES [dbo].[EmailTemplate] ([EmailTemplateID])

GO
