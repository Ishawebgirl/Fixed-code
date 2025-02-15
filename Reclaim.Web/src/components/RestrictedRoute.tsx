import React from 'react'
import { Navigate } from 'react-router-dom'
import useIdentity from 'hooks/useIdentity'

interface RestrictedRouteProps {
  element: React.ReactElement
  role: string
}

const RestrictedRoute: React.FC<RestrictedRouteProps> = ({ element, role }) => {
  const identity = useIdentity()

  if (identity?.role === role) {
    return element
  } else {
    return (
      <Navigate
        to={'/signin?redirectTo=' + encodeURIComponent(window.location.pathname)}
      />
    )
  }
}

export default RestrictedRoute
