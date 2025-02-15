ALTER TABLE [dbo].[AccountAuthentication] WITH CHECK ADD CONSTRAINT [FK_AccountAuthentication_Account]
   FOREIGN KEY([AccountID]) REFERENCES [dbo].[Account] ([AccountID])

GO
ALTER TABLE [dbo].[AccountAuthentication] WITH CHECK ADD CONSTRAINT [FK_AccountAuthentication_IdentityProvider]
   FOREIGN KEY([IdentityProviderID]) REFERENCES [dbo].[IdentityProvider] ([IdentityProviderID])

GO
