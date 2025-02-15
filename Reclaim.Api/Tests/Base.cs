using Xunit;
using RestSharp;
using Newtonsoft.Json;
using RestSharp.Authenticators;
using System;
using System.Dynamic;
using System.Collections.Generic;
using System.Linq;
using System.Configuration;

namespace Reclaim.Api.Tests;

public abstract class Base
{
    protected static RestClient _restClient = new RestClient();

    protected dynamic RequestWithJwtAccessToken(string url, RestSharp.Method method, dynamic parameters, string accessToken)
    {
        var request = new RestRequest(url, method);
        request.AddHeader("Authorization", $"Bearer {accessToken}");

        if (parameters != null)
        {
            request.AddHeader("Content-Type", "application/json");
            request.AddBody(parameters as object);
        }

        var response = _restClient.Execute(request);
        var json = response.Content ?? "";
        dynamic data = JsonConvert.DeserializeObject<ExpandoObject>(json);
        data!.statusCode = (int)response.StatusCode;

        return data;
    }

    protected dynamic RequestWithJwtAccessToken(string url, RestSharp.Method method, dynamic? parameters, string emailAddress = Default.CustomerEmailAddress, string password = Default.CustomerPassword)
    {
        var token = GetJwtAccessToken(emailAddress, password);

        return RequestWithJwtAccessToken(url, method, parameters, token.accessToken);
    }

    protected dynamic RequestWithoutJwtAccessToken(string url, Method method, dynamic? parameters = null)
    {
        var request = new RestRequest(url, method);

        if (parameters != null)
        {
            request.AddHeader("Content-Type", "application/json");
            request.AddBody(parameters as object);
        }

        request.AddHeader("Accept", "application/json");

        var response = _restClient.Execute(request);
        var json = response.Content ?? "";
        dynamic data = JsonConvert.DeserializeObject<ExpandoObject>(json);

        if (data == null)
            data = new ExpandoObject();

        data.statusCode = (int)response.StatusCode;

        return data;
    }

    protected dynamic GetJwtAccessToken(string emailAddress = Default.CustomerEmailAddress, string password = Default.CustomerPassword, string? refreshToken = null, string? accessToken = null)
    {
        RestRequest request;

        if (refreshToken == null)
        {
            request = new RestRequest("accounts/authenticate", Method.Post);
            request.AddBody(new { emailAddress = emailAddress, password = password });
        }
        else
        {
            request = new RestRequest("accounts/authenticate/refresh", Method.Post);
            request.AddBody(new { emailAddress = emailAddress, refreshToken = refreshToken });
            request.AddHeader("Authorization", $"Bearer {accessToken}");
        }

        request.AddHeader("Content-Type", "application/json");
        request.AddHeader("Accept", "application/json");

        var response = _restClient.Execute(request);

        var json = response.Content ?? "";
        dynamic data = JsonConvert.DeserializeObject<ExpandoObject>(json);
        data!.statusCode = (int)response.StatusCode;

        switch (data.statusCode)
        {
            case 200:
                CustomAssert.ThatObjectHasProperty(data, "accessToken");
                CustomAssert.ThatObjectDoesNotHaveProperty(data, "errorCode");
                break;

            case 401:
            case 403:
            case 429:
                CustomAssert.ThatObjectDoesNotHaveProperty(data, "accessToken");
                CustomAssert.ThatObjectHasProperty(data, "errorCode");
                break;

            case 400:
                // unable to trap bad JWT refresh token
                break;

            default:
                Assert.Fail("Invalid status code");
                break;
        }

        return data;
    }
}