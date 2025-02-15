ALTER TABLE [dbo].[Customer] WITH CHECK ADD CONSTRAINT [FK_Customer_Account]
   FOREIGN KEY([AccountID]) REFERENCES [dbo].[Account] ([AccountID])

GO
ALTER TABLE [dbo].[Customer] WITH CHECK ADD CONSTRAINT [FK_Customer_CustomerStatus]
   FOREIGN KEY([CustomerStatusID]) REFERENCES [dbo].[CustomerStatus] ([CustomerStatusID])

GO
ALTER TABLE [dbo].[Customer] WITH CHECK ADD CONSTRAINT [FK_Customer_State]
   FOREIGN KEY([StateID]) REFERENCES [dbo].[State] ([StateID])

GO
