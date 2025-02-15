ALTER TABLE [dbo].[Administrator] WITH CHECK ADD CONSTRAINT [FK_Administrator_Account]
   FOREIGN KEY([AccountID]) REFERENCES [dbo].[Account] ([AccountID])

GO
