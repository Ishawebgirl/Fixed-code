CREATE TABLE [dbo].[Cache] (
   [ID] [nvarchar](449) NOT NULL,
   [Value] [varbinary](max) NOT NULL,
   [ExpiresAtTime] [datetimeoffset] NOT NULL,
   [SlidingExpirationInSeconds] [bigint] NULL,
   [AbsoluteExpiration] [datetimeoffset] NULL

   ,CONSTRAINT [PK__Cache__3214EC276C2898CD] PRIMARY KEY CLUSTERED ([ID])
)

CREATE NONCLUSTERED INDEX [Index_ExpiresAtTime] ON [dbo].[Cache] ([ExpiresAtTime])

GO
