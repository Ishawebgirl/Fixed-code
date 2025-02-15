ALTER TABLE [dbo].[Chat] WITH CHECK ADD CONSTRAINT [FK_Chat_Account]
   FOREIGN KEY([AccountID]) REFERENCES [dbo].[Account] ([AccountID])

GO
ALTER TABLE [dbo].[Chat] WITH CHECK ADD CONSTRAINT [FK_Chat_ChatType]
   FOREIGN KEY([ChatTypeID]) REFERENCES [dbo].[ChatType] ([ChatTypeID])

GO
ALTER TABLE [dbo].[Chat] WITH CHECK ADD CONSTRAINT [FK_Chat_Claim]
   FOREIGN KEY([ClaimID]) REFERENCES [dbo].[Claim] ([ClaimID])

GO
