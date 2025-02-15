import { LandingLayout } from 'layouts/LandingLayout'
import { Link } from 'react-router-dom'

const Index = () => {
  return (
    <LandingLayout id="index" image="images/landing.jpg">
      <div className="logo-container">
        <img className="logo" src="images/logo-white.png" alt="Logo"></img>
        <Link className="styled-button signin-button" to={'/signin'}>
          Sign in
        </Link>
      </div>
      <div className="container">
        <div className="content">
          <div className="title">
            Property insurance SIU
            <br />
            And fraud recovery
          </div>
          <div className="subtitle">
            Contact us for a free AI-powered analysis of your claims database.
          </div>
          <div className="button-row">
            <Link className="styled-button" to={'/register'}>
              Learn More
            </Link>
          </div>
        </div>
      </div>
    </LandingLayout>
  )
}

export default Index
