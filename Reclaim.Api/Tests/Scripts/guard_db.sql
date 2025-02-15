if @@servername != 'hammer-pc' begin
    raiserror('Invalid hostname', 16, 1)
    return
end

if db_name() != 'reclaim' begin
    raiserror('Invalid database', 16, 1)
    return
end
