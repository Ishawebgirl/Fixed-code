import 'assets/styles/globals.scss'
import 'assets/styles/fonts.scss'

import { AuthenticationProvider } from '../contexts/AuthenticationContext'
import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { Role } from 'api/model'
import Index from './public/Index'
import ConfirmAccount from './public/ConfirmAccount'
import ForgotPassword from './public/ForgotPassword'
import Register from './public/Register'
import SetPassword from './public/SetPassword'
import SignIn from './public/SignIn'
import ThankYou from './public/ThankYou'

import Claims from './Claims'
import Claim from './Claim'
import ClaimDocuments from './ClaimDocuments'
import ClaimChat from './ClaimChat'

import Administrators from './Administrators'
import Investigators from './Investigators'
import Customers from './Customers'
import SignIns from './Signins'
import Jobs from './Jobs'

import Dashboard from './Dashboard'

import Privacy from './public/Privacy'

import { AuthenticatedLayout } from '../layouts/AuthenticatedLayout'
import RestrictedRoute from 'components/RestrictedRoute'

// prettier-ignore
const App = () => {
  return (
    <AuthenticationProvider>
      <Routes>
        <Route path="/" element={<Index />} />
        <Route path="/confirmaccount" element={<ConfirmAccount />} />
        <Route path="/forgotpassword" element={<ForgotPassword />} />
        <Route path="/privacy" element={<Privacy />} />
        <Route path="/register" element={<Register />} />
        <Route path="/setpassword" element={<SetPassword />} />
        <Route path="/signin" element={<SignIn />} />
        <Route path="/thankyou" element={<ThankYou />} />

        <Route path="/" element={<AuthenticatedLayout />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/claims" element={<Claims />}></Route>
          <Route path="/claims/:uniqueID" element={<Claim />}></Route>          
          <Route path="/claims/:uniqueID/chats" element={<ClaimChat />}></Route>
          <Route path="/claims/:uniqueID/chats/:chatUniqueID" element={<ClaimChat />}></Route>
          <Route path="/claims/:uniqueID/documents" element={<ClaimDocuments />}></Route>
          <Route path="/claims/:uniqueID" element={<Claim />}></Route>
          <Route path="/claims/:uniqueID/documents" element={<ClaimDocuments />}></Route>
          <Route path="/investigators" element={<Investigators />}></Route>
          <Route path="/customers" element={<Customers />}></Route>
          <Route path="/administrators" element={ <RestrictedRoute role={Role.Administrator} element={<Administrators />} /> } />
          <Route path="/signins" element={<RestrictedRoute role={Role.Administrator} element={<SignIns />} />} />
          <Route path="/jobs" element={<RestrictedRoute role={Role.Administrator} element={<Jobs />} />} />
        </Route>
      </Routes>
    </AuthenticationProvider>
  )
}

export default App
