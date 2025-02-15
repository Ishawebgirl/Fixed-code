using System;
using System.Data;
using System.Configuration;
using System.Collections.Generic;
using System.Xml;
using System.IO;
using System.Reflection;
using Microsoft.Data.SqlClient;

namespace Reclaim.Api.Tests;

internal sealed class SqlHelper
{
    private static string? _connectionString;

    internal static DataTable ExecuteDataTable(string commandText, params SqlParameter[] parameters)
    {
        var dataSet = ExecuteDataset(commandText, parameters);

        return dataSet.Tables[0];
    }

    internal static IEnumerable<dynamic> ExecuteDataTableDynamic(string commandText, params SqlParameter[] parameters)
    {
        var dataSet = ExecuteDataset(commandText, parameters);

        return dataSet.Tables[0].AsDynamicEnumerable();
    }

    internal static DataSet ExecuteDataset(string commandText, params SqlParameter[] parameters)
    {
        using (var connection = CreateConnection())
        {
            var command = CreateCommand(CommandType.Text, commandText, connection, parameters);
            var adapter = CreateDataAdapter(command);
            var ds = new DataSet();
            adapter.Fill(ds);

            return ds;
        }
    }

    internal static object ExecuteScalar(string commandText, params SqlParameter[] parameters)
    {
        using (SqlConnection connection = CreateConnection())
        {
            var command = CreateCommand(CommandType.Text, commandText, connection, parameters);
            connection.Open();

            return command.ExecuteScalar();
        }
    }

    internal static int ExecuteNonQuery(string commandText, params SqlParameter[] parameters)
    {
        using (SqlConnection connection = CreateConnection())
        {
            var command = CreateCommand(CommandType.Text, commandText, connection, parameters);
            connection.Open();

            return command.ExecuteNonQuery();
        }
    }

    internal static string GetConnectionString()
    {
        if (_connectionString != null)
            return _connectionString;

        var path = Assembly.GetExecutingAssembly().Location;
        var root = path.Substring(0, path.IndexOf("\\bin\\"));

        AppDomain.CurrentDomain.SetData("APP_CONFIG_FILE", $"{root}/Settings/Local.config");
        _connectionString = ConfigurationManager.AppSettings["ConnectionString"]!;

        return _connectionString;
    }

    internal static SqlConnection CreateConnection()
    {
        return new SqlConnection(GetConnectionString());
    }

    internal static SqlCommand CreateCommand(CommandType commandType, string commandText, SqlConnection connection, params SqlParameter[] parameters)
    {
        var file = File.ReadAllText("../../../Scripts/guard_db.sql");

        var command = new SqlCommand(file + "\r\n\r\n" + commandText);

        command.CommandTimeout = 0;
        command.Connection = connection;
        command.CommandType = commandType;

        if (parameters != null && parameters.Length > 0)
            command.Parameters.AddRange(parameters);

        return command;
    }

    internal static SqlDataAdapter CreateDataAdapter(SqlCommand command)
    {
        return new SqlDataAdapter(command);
    }

    internal static SqlParameter GetSqlParameter(string column, object value)
    {
        if (value == null)
            return new SqlParameter(column, DBNull.Value);
        else
            return new SqlParameter(column, value);
    }

    internal static void AddSqlParameter(List<SqlParameter> dataParams, string name, object value)
    {
        if (value != null)
            dataParams.Add(new SqlParameter(name, value));
    }

    internal static void AddSqlParameter(List<SqlParameter> dataParams, string name, object value, ParameterDirection direction)
    {
        var output = new SqlParameter(name, value);

        output.Direction = direction;

        dataParams.Add(output);
    }
}
