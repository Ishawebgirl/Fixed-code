ALTER TABLE [dbo].[AccountPassword] WITH CHECK ADD CONSTRAINT [FK_AccountPassword_Account]
   FOREIGN KEY([AccountID]) REFERENCES [dbo].[Account] ([AccountID])

GO
